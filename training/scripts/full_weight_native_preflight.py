"""Exercise production native NCCL publication inside an existing allocation.

Launch scripts/launch_platypus_server.sh on GPU 0 with SERVER_WEIGHT_MODE=full_weight,
SERVER_ENVIRONMENT=arc2, SERVER_TENSOR_PARALLEL=1, the pinned HF model/revision,
and the production vLLM 0.25.1 environment. Once its health endpoint is ready:

CUDA_VISIBLE_DEVICES=1 "$PLATYPUS_VENV/bin/python" -m \
    scripts.full_weight_native_preflight --base-url http://127.0.0.1:8000 \
    --output-dir "$SCRATCH_DIR/native-full-weight-preflight"

The sender streams cached BF16 language tensors, never a whole GPU model. Two
publications of identical source bytes must preserve multimodal draw evidence.
This verifies serving/publication, not FSDP gradients or trainer/server parity.
Encoder tensors are omitted; the production HTTP API has no weight-inspection
RPC, so this gate cannot attest that the engine's encoder bytes stayed unchanged.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback
import uuid

import httpx
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from training.policy_sync import (
    VllmControlClient,
    VllmNcclTrainerEndpoint,
    VllmNcclWeightTransfer,
    WeightManifest,
)
from training.run_artifacts import write_json_atomic


MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
# Identical BF16 source, single engine, identical greedy request and cold cache:
# permit only FP32 log-softmax rounding, not the broader HF/vLLM parity budget.
LOGPROB_ATOL = 1e-5
SAMPLING = {
    "n": 1,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "max_tokens": 16,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
    "min_p": 0.0,
    "min_tokens": 0,
    "logprobs": 0,
    "skip_special_tokens": False,
    "ignore_eos": False,
    "seed": 1729,
}


class CachedLanguageSource:
    """Validate all shard headers, then stage one canonical tensor at a time."""

    def __init__(self):
        index = Path(hf_hub_download(
            MODEL, "model.safetensors.index.json", revision=REVISION,
            local_files_only=True,
        ))
        self.root = index.parent
        self.weight_map = json.loads(index.read_text())["weight_map"]
        shapes = {}
        self.excluded_vision = []
        self.excluded_mtp = []
        for shard in sorted(set(self.weight_map.values())):
            expected = {name for name, file in self.weight_map.items() if file == shard}
            with safe_open(self.root / shard, framework="pt", device="cpu") as source:
                if set(source.keys()) != expected:
                    raise RuntimeError(f"Safetensors header/index names differ: {shard}")
                for name in sorted(expected):
                    header = source.get_slice(name)
                    if name.startswith("model.visual."):
                        self.excluded_vision.append(name)
                        continue
                    # Both HF's Qwen3_5PreTrainedModel and the pinned vLLM
                    # conditional-generation loader exclude this auxiliary head.
                    if name.startswith("mtp."):
                        self.excluded_mtp.append(name)
                        continue
                    if not name.startswith(("model.language_model.", "lm_head.")):
                        raise RuntimeError(f"Unexpected pinned model tensor: {name}")
                    if header.get_dtype() != "BF16":
                        raise RuntimeError(f"Pinned language tensor is not BF16: {name}")
                    shapes[name] = tuple(header.get_shape())
        if not self.excluded_vision or not {
            "model.language_model.embed_tokens.weight", "lm_head.weight"
        } <= shapes.keys():
            raise RuntimeError("Expected complete language and frozen vision namespaces")
        names = tuple(sorted(shapes))
        self.manifest = WeightManifest(
            names, ("bfloat16",) * len(names), tuple(shapes[name] for name in names)
        )
        self.content_sha256 = None

    def parameters(self):
        self.content_sha256 = None
        digest = hashlib.sha256()

        def materialize():
            for name in self.manifest.names:
                with safe_open(
                    self.root / self.weight_map[name], framework="pt", device="cpu"
                ) as source:
                    cpu = source.get_tensor(name).contiguous()
                # Hash the actual transfer bytes independently on both passes.
                digest.update(name.encode() + b"\0")
                digest.update(cpu.view(torch.uint8).numpy().tobytes())
                tensor = cpu.to(device="cuda:0", dtype=torch.bfloat16)
                yield name, tensor
                del tensor, cpu

        yield from self.manifest.validate_parameters(materialize())
        self.content_sha256 = digest.hexdigest()


async def draw(proxy, prompt, label):
    from ullm.backends.engine_proxy import serialize_output

    outputs = [output async for output in proxy.generate(
        prompt, SAMPLING, f"native-preflight#{label}-{uuid.uuid4().hex}"
    )]
    if len(outputs) != 1:
        raise RuntimeError("Expected one native engine output")
    return serialize_output(outputs[0])


def multimodal_prompt(proxy):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (128, 128), "blue")
    ImageDraw.Draw(image).rectangle((0, 0, 63, 127), fill="red")
    processor = proxy.get_processor()
    text = processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Name the two colors in this image."},
        ]}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    prompt = {
        "prompt_token_ids": inputs["input_ids"][0].tolist(),
        "multi_modal_data": {"image": [image]},
        "mm_processor_kwargs": proxy.metadata["mm_processor_kwargs"],
    }
    return {
        "prompt_token_ids": prompt["prompt_token_ids"],
        "media": proxy.prepare_media(prompt),
    }


async def run(args, report):
    from training.engine_protocol import unpack_output
    from ullm.backends.engine_proxy import EngineProxy

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run inside an existing Slurm allocation")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose only the sender GPU with CUDA_VISIBLE_DEVICES=1")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("The diagnostic sender is a single process, not torchrun")
    if importlib.metadata.version("vllm") != "0.25.1":
        raise RuntimeError("Use the production pinned vLLM 0.25.1 environment")
    torch.cuda.set_device(0)
    report["sender_gpu"] = torch.cuda.get_device_name(0)
    report["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    api_key = os.environ.get("SERVER_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    control = VllmControlClient(args.base_url, auth_headers=headers)
    proxies = []
    endpoint = VllmNcclTrainerEndpoint(host="127.0.0.1")
    pinned_snapshot = Path(hf_hub_download(
        MODEL, "model.safetensors.index.json", revision=REVISION,
        local_files_only=True,
    )).parent.resolve()
    completed = False

    async def connect():
        proxy = EngineProxy(base_url=args.base_url, model=MODEL, api_key=api_key)
        proxies.append(proxy)
        await proxy.connect()
        # Offline vLLM resolves the HF ID to this exact cached snapshot before
        # constructing ModelConfig, retaining the revision and served alias.
        model_id = proxy.metadata["model_id"]
        if (
            (model_id != MODEL and Path(model_id).resolve() != pinned_snapshot)
            or proxy.metadata["model_revision"] != REVISION
            or proxy.metadata["served_model_name"] != MODEL
            or proxy.metadata["vllm_version"] != "0.25.1"
        ):
            raise RuntimeError("Native server differs from the pinned HF model/revision")
        return proxy

    try:
        unpublished = await connect()
        if "weight_version" in unpublished.metadata["adapter"]:
            raise RuntimeError("Preflight requires a fresh, unpublished native engine")
        if control.world_size() != 1:
            raise RuntimeError("The two-GPU diagnostic requires one native engine worker")
        prompt = multimodal_prompt(unpublished)
        write_json_atomic(args.output_dir / "request.json", {
            "prompt": prompt, "sampling_params": SAMPLING,
        })
        try:
            await draw(unpublished, prompt, "unpublished")
        except httpx.HTTPStatusError as error:
            if error.response.status_code != 400 or (
                "requires a published policy" not in error.response.text
            ):
                raise
            report["unpublished_refused"] = True
        else:
            raise RuntimeError("Native engine admitted an unpublished policy")
        # Closing a failed proxy repeats its recorded error by design.
        try:
            await unpublished.aclose()
        except httpx.HTTPStatusError:
            pass
        proxies.remove(unpublished)

        source = CachedLanguageSource()
        report["manifest_sha256"] = source.manifest.sha256
        report["language_tensor_count"] = len(source.manifest.names)
        report["excluded_vision_tensor_count"] = len(source.excluded_vision)
        report["excluded_mtp_tensor_count"] = len(source.excluded_mtp)
        write_json_atomic(args.output_dir / "manifest.json", {
            **source.manifest.update_info(packed=True),
            "excluded_vision_names": source.excluded_vision,
            "excluded_mtp_names": source.excluded_mtp,
        })
        transfer = VllmNcclWeightTransfer([control], trainer=endpoint)
        policies, outputs, hashes = [], [], []
        report["publications"] = []
        for index in (1, 2):
            print(f"Publishing version {index}: {len(source.manifest.names)} BF16 tensors", flush=True)
            started = time.monotonic()
            publication = transfer.transfer(
                source.manifest,
                send_weights=lambda: endpoint.send_weights(source.parameters()),
                weight_version=uuid.uuid4().hex,
            )
            if source.content_sha256 is None:
                raise RuntimeError("NCCL sender did not exhaust the source tensor stream")
            proxy = await connect()
            if proxy.metadata["adapter"] != control.adapter_identity:
                raise RuntimeError("Engine metadata differs from publication acknowledgement")
            policies.append(proxy)
            output = await draw(proxy, prompt, f"v{index}")
            write_json_atomic(args.output_dir / f"generation_v{index}.json", output)
            evidence = unpack_output(output)
            if not evidence["output_token_ids"]:
                raise RuntimeError("Native multimodal generation returned no tokens")
            if evidence["num_cached_tokens"] not in (None, 0):
                raise RuntimeError("First draw after publication reused cached tokens")
            outputs.append(evidence)
            hashes.append(source.content_sha256)
            report["publications"].append({
                **publication, "source_content_sha256": source.content_sha256,
                "seconds": time.monotonic() - started,
                "adapter": control.adapter_identity,
            })
            write_json_atomic(args.output_dir / "report.json", report)
        if not hashes[0] or hashes[0] != hashes[1]:
            raise RuntimeError("The two publications did not send identical source bytes")
        for key in ("prompt_token_ids", "output_token_ids", "media_evidence", "text"):
            if outputs[0][key] != outputs[1][key]:
                raise RuntimeError(f"Same-source native generations differ in {key}")
        before, after = [torch.tensor(value["output_logprobs"]) for value in outputs]
        torch.testing.assert_close(before, after, rtol=0.0, atol=LOGPROB_ATOL)
        report["same_source_parity"] = {
            "tokens_exact": True, "media_exact": True,
            "max_logprob_abs_difference": float((before - after).abs().max()),
            "logprob_atol": LOGPROB_ATOL,
        }
        report["sender_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        try:
            await draw(policies[0], prompt, "stale-v1")
        except ValueError as error:
            if "identity changed after connection" not in str(error):
                raise
            report["stale_proxy_refused"] = True
        else:
            raise RuntimeError("A retained version-1 proxy accepted version-2 evidence")
        try:
            await policies[0].aclose()
        except ValueError:
            pass
        proxies.remove(policies[0])
        completed = True
    finally:
        for proxy in proxies:
            try:
                await proxy.aclose()
            except Exception:
                if completed:
                    raise
        # Failed NCCL operations are terminal; do not add teardown collectives.
        if completed:
            endpoint.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running", "model": MODEL, "revision": REVISION,
        "scope": "native NCCL serving; identical cached BF16 language weights",
        "limitations": [
            "Vision tensors are excluded from transfer; engine encoder bytes are not inspected.",
            "Stale identity is rejected by EngineProxy after the draw, not by a native expected-version field.",
            "This does not validate FSDP training, optimizer updates, or HF/vLLM logprob parity.",
        ],
    }
    write_json_atomic(args.output_dir / "report.json", report)
    try:
        asyncio.run(run(args, report))
        report["status"] = "passed"
    except BaseException:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        write_json_atomic(args.output_dir / "report.json", report)


if __name__ == "__main__":
    main()
