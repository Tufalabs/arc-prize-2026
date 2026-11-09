"""Render recorded OpenAI-API seal traces (e.g. Kimi K3.1) into e307's serving token ids (CPU, no engine).

python scripts/flash_mixed_cosine120/conversion/kimi_steps.py kimi --traces DIR --run-dir JOB_DIR \
    --serving STEPS_JSONL --run-config RUN_CONFIG --checkpoint CKPT --out OUT --summary-thinking \
    [--workers N] [--skip-unparseable-calls]
python scripts/flash_mixed_cosine120/conversion/kimi_steps.py check --traces E307_TRACES \
    --run-config RUN_CONFIG --checkpoint CKPT --out OUT RUN_ID ...

A recorded step stores the request conversation in the provider's message shape and no token ids (OpenAI chat:
``tool_calls`` on the assistant, results as role ``tool``). This rebuilds what e307's in-process backend (ullm
vllmx, ``raw_tool_calls``) would have sent for the same conversation, then the ids exactly as that backend sends
them under exact history:

- messages: system and user as recorded (image parts become ``{"type": "image"}`` placeholders, the frames are
  read from the trace's media); assistant turns keep ``reasoning_content`` and carry their calls as structured
  ``tool_calls`` (the chat template writes the ``<tool_call>`` XML); a ``tool`` result becomes the user message
  ``[Tool result for NAME]\\n...`` (`` [ERROR]`` when the call failed, read from the run's diagnostics spans).
- completion: the template's render of the history plus the answer, minus its render with the generation
  prompt (it opens ``<think>``), without the framing newline; tokenized standalone and checked to decode back.
- prompt: the first call of a history renders afresh through the image processor; every later call is
  ``ullm.continuation.continue_prompt`` over the previous prompt + completion (the backend's own continuation),
  and renders afresh only where the recorded history was rewritten (context summaries).
- context summaries: ``--summary-thinking`` preserves recorded reasoning before the summary, matching
  the current Kimi-student profile. ``--no-summary-thinking`` injects ``\n</think>\n\n`` after the
  generation prompt and drops summary reasoning. Preparation passes the actual student profile value;
  ``check`` uses each step's recorded ``sampling_params`` injection.

``--skip-unparseable-calls``: a step whose answer or history holds a tool call whose ``arguments`` are not JSON
(an answer cut off by ``max_tokens`` inside a call, kept in the history that follows it) has no template render; it
is left out and listed in the receipt, and the next kept step renders afresh. Without the flag such a step fails.

``check`` is the converter's test on e307's own traces: their messages are rewritten into the OpenAI shape
(``<tool_call>`` XML parsed into ``tool_calls``, ``[Tool result for X]`` user turns into ``tool`` messages, the
error flag kept aside the way an OpenAI trace loses it), then rendered by the same code, continuing over the
recorded output ids, and compared with the recorded ``input_token_ids`` (and the rendered completion with the
recorded ``output_text``/``output_token_ids``). Both modes also parse every rendered completion with e307's
tool-call parser and compare it with the recorded calls.
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import zipfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

RESULT = re.compile(r"\A\[Tool result for ([^\]\s]+)( \[ERROR\])?\]\n", re.DOTALL)
CALL = re.compile(r"<tool_call>\n<function=([^>\n]+)>\n(.*?)</function>\n</tool_call>", re.DOTALL)
PARAM = re.compile(r"<parameter=([^>\n]+)>\n(.*?)\n</parameter>\n", re.DOTALL)

# seal/prompts.py: the context-summary request; seal/agent.py _SUMMARY_THINK_CLOSE (no summary_thinking).
SUMMARY_MARKER = "Your context window is nearly full. Write a "
SUMMARY_THINK_CLOSE = "\n</think>\n\n"

TOK = PROC = PARSER = None
G = {}


def setup(checkpoint, mm_kwargs, tools):
    """Tokenizer, image processor and tool parser as e307's backend builds them (per worker)."""
    global TOK, PROC, PARSER
    import transformers
    from vllm.tool_parsers import ToolParserManager

    # max_pixels acts only at construction: a Qwen processor ignores it per call (vLLM splits the kwargs that way).
    PROC = transformers.AutoProcessor.from_pretrained(checkpoint, **mm_kwargs)
    TOK = PROC.tokenizer
    PARSER = ToolParserManager.get_tool_parser("qwen3_coder")(TOK)
    G.update(mm_kwargs=dict(mm_kwargs), tools=tools)


def encoder(images):
    """The backend's delta encoder: text through the processor with the images whose placeholders it carries."""
    from PIL import Image

    def encode(text):
        if text.count(str(PROC.image_token)) != len(images):
            raise ValueError("image placeholders and images do not line up")
        if not images:
            return [int(t) for t in PROC.tokenizer(text, truncation=False, max_length=None)["input_ids"]]
        pils = [Image.open(p).convert("RGB") for p in images]
        ids = PROC(text=[text], images=pils, truncation=False, max_length=None, **G["mm_kwargs"])["input_ids"]
        first = ids[0] if not isinstance(ids[0], int) else ids
        return [int(t) for t in (first.tolist() if hasattr(first, "tolist") else first)]

    return encode


def render(messages, generation):
    return str(TOK.apply_chat_template(messages, add_generation_prompt=generation, tokenize=False, **G["template"]))


def assistant(message):
    """An OpenAI-shape assistant message as the template's message (calls structured, arguments as objects)."""
    out = {"role": "assistant", "content": message.get("content") or ""}
    reasoning = (message.get("reasoning_content") or "").strip()
    if reasoning:
        out["reasoning_content"] = reasoning
    if message.get("tool_calls"):
        out["tool_calls"] = [
            {"type": "function", "function": {"name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"])}}
            for c in message["tool_calls"]
        ]
    return out


# The line Kimi's seal prompt adds over e307's serving prompt, in the system prompt and in the user messages that open a
# game or follow a context reset (--drop-world-model-line).
WORLD_MODEL_LINE = (
    "- Your default mode should be to use a world model; in general, at least attempt to write one, "
    "and only stop if you are convinced it's not useful on this game.\n"
)


def user_text(text):
    """A user message's text as rendered: without the world-model line under --drop-world-model-line."""
    return text.replace(WORLD_MODEL_LINE.removeprefix("- "), "") if G.get("drop_world_model_line") else text


def to_e307(messages, trace, errors):
    """OpenAI-shape messages -> the e307 backend's template messages + the frames they show, in order."""
    out, images, names = [], [], {}
    for m in messages:
        role = m["role"]
        if role == "system":
            assert isinstance(m["content"], str)
            content = m["content"]
            if G.get("drop_world_model_line"):
                assert content.count(WORLD_MODEL_LINE) == 1, "the world-model line is not in the system prompt exactly once"
                content = content.replace(WORLD_MODEL_LINE, "")
            out.append({"role": "system", "content": content})
        elif role == "user":
            content = m["content"]
            if isinstance(content, str):
                out.append({"role": "user", "content": user_text(content)})
                continue
            blocks = []
            for part in content:
                if part["type"] == "text":
                    blocks.append({"type": "text", "text": user_text(part["text"])})
                elif part["type"] == "image_url":
                    blocks.append({"type": "image"})
                    url = part["image_url"]["url"]
                    assert url.startswith("media/") and ".." not in url, url
                    images.append(str(trace / url))
                else:
                    raise ValueError(f"unsupported user part {part['type']}")
            out.append({"role": "user", "content": blocks})
        elif role == "assistant":
            out.append(assistant(m))
            for c in m.get("tool_calls") or []:
                names[c["id"]] = c["function"]["name"]
        elif role == "tool":
            tag = " [ERROR]" if errors[m["tool_call_id"]] else ""
            out.append({"role": "user", "content": f"[Tool result for {names[m['tool_call_id']]}{tag}]\n{m['content']}"})
        else:
            raise ValueError(f"unsupported role {role}")
    return out, images


def parse_calls(text):
    """e307's tool-call parser over a completion's answer (after the think block): [(name, arguments)]."""
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
        ChatCompletionToolsParam,
        FunctionDefinition,
    )

    answer = text.split("</think>", 1)[1] if "</think>" in text else text
    answer = answer.removesuffix(str(TOK.eos_token)).strip()
    params = [ChatCompletionToolsParam(type="function", function=FunctionDefinition(**t["function"])) for t in G["tools"]]
    got = PARSER.extract_tool_calls(answer, request=ChatCompletionRequest(model="", messages=[], tools=params))
    if not got.tools_called:
        return []
    return [(c.function.name, json.loads(c.function.arguments)) for c in got.tool_calls]


def text_of(message):
    content = message.get("content")
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content or [] if p.get("type") == "text")


def summary_injection(step):
    """Select the summary prefill from the explicit student profile switch."""
    last = (step.get("messages") or [{}])[-1]
    if not G.get("summary_thinking", False) and last.get("role") == "user" and SUMMARY_MARKER in text_of(last):
        return SUMMARY_THINK_CLOSE
    return None


def convert(steps, trace, errors, witness_outputs=None, injections=None):
    """Per step: prompt ids, completion ids/text, images, whether it continued, and the parse-back result.

    ``errors``: call id -> failed, or one such dict per step. ``witness_outputs``: the ids each answer was
    continued over (default: the rendered completion; ``check`` passes the recorded samples). ``injections``:
    per step the text served as ids after the generation prompt (default: ``summary_injection``)."""
    import ullm.continuation as continuation

    served, last, served_images = None, None, []
    for k, step in enumerate(steps):
        messages, images = to_e307(step["messages"], trace, errors[k] if isinstance(errors, list) else errors)
        injection = summary_injection(step) if injections is None else injections[k]
        answer = assistant(step["output_message"])
        if injection == SUMMARY_THINK_CLOSE:
            # Served on the closed think block: no reasoning in the answer.
            answer.pop("reasoning_content", None)
        elif injection is not None:
            raise ValueError(f"step {k}: unsupported injection {injection!r}")
        head, answered = render(messages, True), render(messages + [answer], False)
        eos = str(TOK.eos_token)
        if not answered.startswith(head + (injection or "")) or not answered.endswith(eos + continuation.FRAMING):
            raise ValueError(f"step {k}: the answer does not render as a suffix of its prompt")
        completion = answered[len(head) + len(injection or "") : -len(continuation.FRAMING)]
        output = [int(t) for t in TOK.encode(completion, add_special_tokens=False)]
        if TOK.decode(output) != completion:
            raise ValueError(f"step {k}: the completion does not tokenize standalone")
        continued = None
        if last is not None and images[: len(served_images)] == served_images:
            continued = continuation.continue_prompt(
                TOK, served, last, head, encode_delta=encoder(images[len(served_images) :])
            )
        prompt = continued.prompt_ids if continued is not None else encoder(images)(head)
        if injection:
            # As the backend's _injection_ids: tokenized standalone, after the rendered prompt.
            prompt = prompt + [int(t) for t in TOK.encode(injection, add_special_tokens=False)]
        sampled = output if witness_outputs is None else witness_outputs[k]
        last = continuation.witness(
            TOK, epoch=0, prompt_ids=prompt, output_ids=sampled, served_before=served,
            head_render=head, answered_render=answered, image_fingerprints=tuple(images),
            injected=injection or "",
        )
        if last is None:
            raise ValueError(f"step {k}: no witness (the template does not render the answer as a suffix)")
        served, served_images = prompt + sampled, list(images)
        recorded = [(c["function"]["name"], json.loads(c["function"]["arguments"]))
                    for c in step["output_message"].get("tool_calls") or []]
        yield dict(
            step_index=step["step_index"], input_token_ids=prompt, output_token_ids=output, completion=completion,
            images=[os.path.relpath(p, trace) for p in images], continued=continued is not None,
            stop=(step.get("extra") or {}).get("stop_reason"), calls=len(recorded), injection=injection,
            parse_back=parse_calls(completion) == recorded,
        )


def diagnostics_errors(run_dir, run_id, steps):
    """Recover tool error flags from live or archived spans, checking tool-result order."""
    events = {}

    def collect(lines):
        for line in lines:
            e = json.loads(line)
            if e.get("kind") == "tool/result":
                key = (e["t"], e["values"]["tool_name"])
                flag = bool(e["values"]["is_error"])
                if key in events and events[key] != flag:
                    raise ValueError(f"{run_id}: conflicting tool error flags at {key}")
                events[key] = flag

    spans = Path(run_dir) / "games" / run_id / "spans.jsonl"
    if spans.is_file():
        with spans.open() as stream:
            collect(stream)
    for path in sorted(glob.glob(str(Path(run_dir) / "diagnostics-*.zip"))):
        with zipfile.ZipFile(path) as z:
            name = f"games/{run_id}/spans.jsonl"
            if name in z.namelist():
                with z.open(name) as stream:
                    collect(stream)
    ordered = [(name, flag) for (_, name), flag in sorted(events.items())]
    results, names, seen = [], {}, set()
    for step in steps:
        for m in step["messages"]:
            for c in m.get("tool_calls") or []:
                names[c["id"]] = c["function"]["name"]
            if m["role"] == "tool" and m["tool_call_id"] not in seen:
                seen.add(m["tool_call_id"])
                results.append(m["tool_call_id"])
    assert [names[c] for c in results] == [n for n, _ in ordered[: len(results)]], f"{run_id}: spans misaligned"
    return {c: ordered[i][1] for i, c in enumerate(results)}


def unparseable(step):
    """Whether a tool call in the step's answer or history has arguments that are not JSON."""
    for m in step["messages"] + [step["output_message"]]:
        for c in m.get("tool_calls") or []:
            try:
                json.loads(c["function"]["arguments"])
            except json.JSONDecodeError:
                return True
    return False


def segments(steps):
    """The runs of steps with parseable calls (``--skip-unparseable-calls``), and the skipped step indices."""
    if not G.get("skip_unparseable"):
        return [steps], []
    out, skipped, current = [], [], []
    for step in steps:
        if unparseable(step):
            skipped.append(step["step_index"])
            if current:
                out.append(current)
            current = []
        else:
            current.append(step)
    if current:
        out.append(current)
    return out, skipped


def kimi_job(job):
    trace = Path(job["trace"])
    steps = [json.loads(line) for line in (trace / "steps.jsonl").read_text().splitlines()]
    errors = diagnostics_errors(job["run_dir"], trace.name, steps)
    out = Path(job["out"]) / trace.name
    out.mkdir(parents=True)
    n = Counter()
    kept, skipped = segments(steps)
    # Streamed: a run's prompts (whole histories) never sit in memory together.
    with (out / "steps.jsonl").open("w") as f:
        # Each segment starts afresh: no continuation over a skipped step.
        for r in (r for segment in kept for r in convert(segment, trace, errors)):
            f.write(json.dumps(dict(r, run_id=trace.name, game_id=job["game_id"]), separators=(",", ":")) + "\n")
            n["steps"] += 1
            n["continued"] += r["continued"]
            n["parse_back_fail"] += not r["parse_back"]
            n["summaries"] += bool(r["injection"])
            n["completion_tokens"] += len(r["output_token_ids"])
            n["max_sequence"] = max(n["max_sequence"], len(r["input_token_ids"]) + len(r["output_token_ids"]))
    os.symlink(trace / "media", out / "media")
    (out / "_SUCCESS").write_text("")
    extra = dict(skipped_unparseable=skipped) if G.get("skip_unparseable") else {}
    return dict(run_id=trace.name, game_id=job["game_id"], errors=sum(errors.values()), results=len(errors), **n, **extra)


def to_openai(messages, tools):
    """e307 template messages -> the OpenAI shape an API trace records (calls parsed, results as tool messages)."""
    types = {t["function"]["name"]: t["function"]["parameters"].get("properties", {}) for t in tools}
    out, errors, pending, n = [], {}, [], 0
    for m in messages:
        if m["role"] == "assistant":
            content = m.get("content") or ""
            calls, cut = [], content.find("<tool_call>")
            for name, body in CALL.findall(content):
                args = {}
                for key, value in PARAM.findall(body):
                    kind = types.get(name, {}).get(key, {}).get("type", "string")
                    args[key] = value if kind == "string" else json.loads(value)
                n += 1
                calls.append({"id": f"call_{n}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}})
                pending.append(f"call_{n}")
            msg = {"role": "assistant", "content": (content[:cut] if cut >= 0 else content).strip() or None}
            if m.get("reasoning_content"):
                msg["reasoning_content"] = m["reasoning_content"]
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        elif m["role"] == "user" and isinstance(m["content"], str) and RESULT.match(m["content"]):
            head = RESULT.match(m["content"])
            call = pending.pop(0)
            errors[call] = bool(head.group(2))
            out.append({"role": "tool", "tool_call_id": call, "content": m["content"][head.end() :]})
        else:
            out.append(m)
    return out, errors


def check_job(job):
    trace = Path(job["trace"])
    steps = [json.loads(line) for line in (trace / "steps.jsonl").read_text().splitlines()]
    G["template"] = steps[0]["provider_validation"]["request_args"]["chat_template_kwargs"]
    rewritten, errors = [], []
    for s in steps:
        # Call ids are numbered per step's own messages, so each step carries its own flags.
        messages, flags = to_openai(s["messages"], G["template"]["tools"])
        errors.append(flags)
        rewritten.append(dict(s, messages=messages))
    records = list(convert(rewritten, trace, errors, witness_outputs=[s["output_token_ids"] for s in steps],
                           injections=[(s.get("sampling_params") or {}).get("response_injection") for s in steps]))
    rows = []
    for s, r in zip(steps, records):
        rows.append(dict(
            step_index=s["step_index"], input_exact=r["input_token_ids"] == s["input_token_ids"],
            output_text_exact=r["completion"] == s["output_text"], output_ids_exact=r["output_token_ids"] == s["output_token_ids"],
            parse_back=r["parse_back"], continued=r["continued"], recorded_continued=bool((s.get("extra") or {}).get("continued")),
            calls=r["calls"], images=len(r["images"]), stop=r["stop"], injection=r["injection"],
            summary_marker=summary_injection(s) is not None,
            first_input_diff=next((i for i, (a, b) in enumerate(zip(r["input_token_ids"], s["input_token_ids"])) if a != b),
                                  None if len(r["input_token_ids"]) == len(s["input_token_ids"]) else min(len(r["input_token_ids"]), len(s["input_token_ids"]))),
        ))
    return dict(run_id=trace.name, steps=rows, errors_flagged=sum(errors[-1].values()) if errors else 0)


def init(checkpoint, mm_kwargs, tools, template, skip_unparseable=False, drop_world_model_line=False, summary_thinking=False):
    setup(checkpoint, mm_kwargs, tools)
    G["template"] = template
    G["skip_unparseable"] = skip_unparseable
    G["drop_world_model_line"] = drop_world_model_line
    G["summary_thinking"] = summary_thinking


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("kimi", "check"))
    p.add_argument("--traces", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, help="kimi: the recorded job dir (benchmark.json, live or zipped spans)")
    p.add_argument("--serving", type=Path, help="kimi: an e307 steps.jsonl whose chat_template_kwargs are served")
    p.add_argument("--run-config", type=Path, required=True, help="e307 run_config.json (engine mm_processor_kwargs)")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    p.add_argument("--skip-unparseable-calls", action="store_true",
                   help="kimi: leave out steps whose answer or history has a tool call with non-JSON arguments")
    p.add_argument("--drop-world-model-line", action="store_true",
                   help="kimi: render the system prompt without the world-model line, as e307 is served")
    p.add_argument("--summary-thinking", action=argparse.BooleanOptionalAction, default=None,
                   help="kimi: required explicit student profile choice for reasoning before summaries")
    p.add_argument("runs", nargs="*")
    a = p.parse_args()
    if a.mode == "kimi" and a.summary_thinking is None:
        p.error("kimi conversion requires --summary-thinking or --no-summary-thinking from the student profile")
    assert not a.out.exists(), f"{a.out} exists"
    mm_kwargs = json.loads(a.run_config.read_text())["engine_args"]["mm_processor_kwargs"] or {}
    a.out.mkdir(parents=True)
    if a.mode == "check":
        jobs = [dict(trace=str(a.traces / r)) for r in a.runs]
        template = json.loads((a.traces / a.runs[0] / "steps.jsonl").open().readline())["provider_validation"]["request_args"]["chat_template_kwargs"]
        with Pool(min(a.workers, len(jobs)), init, (str(a.checkpoint), mm_kwargs, template["tools"], template)) as pool:
            results = pool.map(check_job, jobs, chunksize=1)
        flat = [s for r in results for s in r["steps"]]
        summary = dict(
            traces=len(results), steps=len(flat), images=sum(s["images"] for s in flat),
            multi_call_steps=sum(s["calls"] > 1 for s in flat), no_call_steps=sum(s["calls"] == 0 for s in flat),
            fresh_renders=sum(not s["continued"] for s in flat),
            injected=sum(bool(s["injection"]) for s in flat),
            summary_marker_agrees=sum(s["summary_marker"] == (s["injection"] == SUMMARY_THINK_CLOSE) for s in flat),
            continued_agrees=sum(s["continued"] == s["recorded_continued"] for s in flat),
            input_exact=sum(s["input_exact"] for s in flat), output_text_exact=sum(s["output_text_exact"] for s in flat),
            output_ids_exact=sum(s["output_ids_exact"] for s in flat), parse_back=sum(s["parse_back"] for s in flat),
            errors_flagged=sum(r["errors_flagged"] for r in results))
        (a.out / "check.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
        (a.out / "CHECK.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary))
        return
    serving = json.loads(a.serving.open().readline())["provider_validation"]["request_args"]["chat_template_kwargs"]
    benchmark = json.loads((a.run_dir / "benchmark.json").read_text())
    games = {r["run_id"]: r["game_id"] for r in benchmark["game_runs"]}
    traces = sorted(p for p in a.traces.iterdir() if (p / "_SUCCESS").is_file())
    first = json.loads((traces[0] / "steps.jsonl").open().readline())
    recorded_tools = [{"type": t["type"], "function": {k: v for k, v in t["function"].items() if k != "strict"}}
                      for t in first["provider_validation"]["request_args"]["tools"]]
    # The traces' own tool schemas (what the model was shown) with e307's serving switches (thinking, effort).
    template = dict(serving, tools=recorded_tools)
    jobs = [dict(trace=str(t), run_dir=str(a.run_dir), out=str(a.out), game_id=games[t.name]) for t in traces]
    with Pool(a.workers, init, (str(a.checkpoint), mm_kwargs, recorded_tools, template, a.skip_unparseable_calls, a.drop_world_model_line, a.summary_thinking)) as pool:
        results = pool.map(kimi_job, jobs, chunksize=1)
    receipt = dict(
        traces=len(results), steps=sum(r["steps"] for r in results), source=str(a.traces), run_dir=str(a.run_dir),
        summary_thinking=a.summary_thinking,
        serving=str(a.serving), checkpoint=str(a.checkpoint), mm_processor_kwargs=mm_kwargs,
        template_kwargs={k: v for k, v in template.items() if k != "tools"}, tools=[t["function"]["name"] for t in recorded_tools],
        tools_equal_serving=recorded_tools == serving.get("tools"),
        tools_sha256=hashlib.sha256(json.dumps(recorded_tools, sort_keys=True).encode()).hexdigest(),
        results=sum(r["results"] for r in results), errors=sum(r["errors"] for r in results),
        continued=sum(r["continued"] for r in results), parse_back_fail=sum(r["parse_back_fail"] for r in results),
        summaries=sum(r["summaries"] for r in results),
        completion_tokens=sum(r["completion_tokens"] for r in results), max_sequence=max(r["max_sequence"] for r in results),
        runs=results)
    if a.drop_world_model_line:
        receipt["dropped_world_model_line"] = WORLD_MODEL_LINE
    if a.skip_unparseable_calls:
        receipt["skipped_unparseable"] = {r["run_id"]: r["skipped_unparseable"] for r in results if r["skipped_unparseable"]}
    (a.out / "STEPS_READY.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "runs"}))


if __name__ == "__main__":
    sys.exit(main())
