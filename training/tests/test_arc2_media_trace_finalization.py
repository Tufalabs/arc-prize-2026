"""The shared native media producer must pass the ARC-2 R5 writer boundary."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest


def _exercise_finalization(trace_root: Path) -> None:
    """Run real producer/collector/validator code without loading a GPU engine."""
    # vllmx imports engine types even when only its trace producer is used.
    # Substitute those unused imports in this child process, never validation.
    def module(name):
        value = ModuleType(name)
        value.__path__ = []
        sys.modules[name] = value
        if "." in name:
            parent, attribute = name.rsplit(".", 1)
            setattr(sys.modules[parent], attribute, value)
        return value

    vllm = module("vllm")
    vllm.__version__ = "0.25.1"
    vllm.SamplingParams = type("SamplingParams", (), {})
    module("vllm.engine")
    module("vllm.engine.async_llm_engine").AsyncLLMEngine = type("AsyncLLMEngine", (), {})
    module("vllm.exceptions").VLLMValidationError = ValueError
    module("vllm.lora")
    module("vllm.lora.request").LoRARequest = type("LoRARequest", (), {})

    from platypus.tracing import TraceCollector
    from taaf.datasets import spec_from_task_json
    from taaf.game import ArcGame, RunSession, Submission
    from taaf.grid import Grid
    from taaf.trace import load_trace
    from test_multimodal_training import fixture_media
    from ullm.backend import GenericSamplingParameters
    from ullm.backends.vllmx import VLLM
    from ullm.conv import Conversation, LLMResponse, StringContent
    from ullm.media import decode_image
    from ullm.tracing import TraceCapability

    _, prompt_ids, media, evidence = fixture_media()
    output_ids = [42, 43]
    tokenizer = SimpleNamespace(
        decode=lambda ids: "answer",
        encode=lambda text, **kwargs: output_ids,
        apply_chat_template=lambda messages, add_generation_prompt, **kwargs: (
            "prefix" if add_generation_prompt else "prefixanswer"
        ),
    )
    backend = VLLM(
        engine=SimpleNamespace(
            get_tokenizer=lambda: tokenizer, model_config=SimpleNamespace(model="model")
        ),
        exact_history=True,
    )
    game = ArcGame(
        spec_data=spec_from_task_json(
            "puzzle",
            {"train": [{"input": [[0]], "output": [[1]]}],
             "test": [{"input": [[0]], "output": [[1]]}]},
        )
    )
    game.start_game(RunSession(trace_root=trace_root))
    run = game.game_run
    collector = TraceCollector(
        TraceCapability(
            provider_name="vllm", provider_version="0.25.1",
            api_kind="engine_proxy", provider_model_id="model",
            validation_method="platypus.vllmx.inprocess_roundtrip.v1",
            model_id="model", tokenizer_id="model", processor_id="model",
        )
    )
    collector.register_run(run)
    request_id = f"{run.run_id}#abc123"
    request_context = {"arc3_progress": {"version": 1, "score": 0.0}}
    element = LLMResponse(final_content=StringContent(text="answer"), stop_reason="stop")
    element._model_contents = SimpleNamespace(
        prompt_token_ids=prompt_ids, num_cached_tokens=0, media_evidence=evidence,
        outputs=[SimpleNamespace(
            token_ids=output_ids,
            logprobs=[{token: SimpleNamespace(logprob=-0.1, rank=1)} for token in output_ids],
        )],
    )
    record = backend._build_trace_step_sync(
        run_id=run.run_id, sampling_params=GenericSamplingParameters(),
        element=element, conversation=Conversation(history_epoch=0),
        llm_input={
            "request_id": request_id, "engine": {"media": media},
            "served_request_context": request_context,
            "trace_ctx": {
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "Solve this puzzle."},
                    {"type": "image"}, {"type": "image"},
                ]}],
                "images": [decode_image(item) for item in media["images"]],
                "template_kwargs": {}, "continued_prompt_ids": prompt_ids,
            },
        },
    )
    collector.record(record)
    correct = Grid.from_lists([[1]])
    game.execute_action(
        Submission(attempts=((correct, correct),)),
        generated_tokens=len(output_ids), uncached_input_tokens=len(prompt_ids),
    )
    game.finish_game()
    folder = collector.finalize(run)
    assert folder is not None and (folder / "_SUCCESS").is_file()
    saved = load_trace(folder).steps[0]
    assert saved.provider_validation["media_reconstruction_status"] == "ok"
    assert saved.provider_validation["original_payload_sha256"] == saved.provider_validation["reconstructed_payload_sha256"]
    assert saved.provider_validation["request_args"]["request_id"] == request_id
    assert saved.provider_validation["request_args"]["request_context"] == request_context
    assert saved.provider_validation["request_args"]["media"] == media
    assert saved.extra["media_evidence"] == evidence
    assert saved.input_token_ids == prompt_ids and saved.output_token_ids == output_ids
    assert saved.logprobs == record.logprobs
    assert saved.action_index == 0 and saved.action == run.history[0].submission
    assert len(saved.media) == 2
    for reference, image in zip(saved.media, record.images, strict=True):
        assert (folder / reference.path).read_bytes() == image.png


def test_native_media_trace_finalizes_through_actual_arc2_writer(tmp_path):
    """Keep ARC-2 and ARC-3 package names isolated in a fresh Python process."""
    root = Path(__file__).resolve().parents[1]
    sources = [
        root / "arc2-platypus/packages/platypus/src",
        root / "arc2-platypus/packages/taaf/src",
        root / "platypus/packages/ullm/src",
    ]
    if not all(path.is_dir() for path in sources):
        pytest.skip("Requires the private ARC-2 and shared ullm submodules")
    if any(importlib.util.find_spec(name) is None for name in ("openai_harmony", "nest_asyncio", "typeguard")):
        pytest.skip("Requires the isolated ARC-2 worker dependencies")
    result = subprocess.run(
        [sys.executable, __file__, str(tmp_path)], check=False, timeout=60,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(map(str, [*sources, root / "tests", root]))},
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _exercise_finalization(Path(sys.argv[1]))
