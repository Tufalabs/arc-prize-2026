from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def generate_report(input_dir: str | Path, output: str | Path | None = None) -> Path:
    input_path = Path(input_dir)
    output_path = (
        Path(output) if output is not None else input_path / "memory_report.html"
    )
    rows = _read_rows(input_path)
    rank_summary = _rank_summary(rows)
    region_summary = _region_summary(rows)
    step_duration_summary = _step_duration_summary(rows)
    step_region_duration_summary = _step_region_duration_summary(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(input_path / "memory_summary.csv", rank_summary)
    _write_csv(input_path / "memory_region_summary.csv", region_summary)
    _write_csv(input_path / "step_duration_summary.csv", step_duration_summary)
    _write_csv(
        input_path / "step_region_duration_summary.csv",
        step_region_duration_summary,
    )
    output_path.write_text(
        _render_html(
            rows,
            rank_summary,
            region_summary,
            step_duration_summary,
            step_region_duration_summary,
        ),
        encoding="utf-8",
    )
    return output_path


def _read_rows(input_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(input_path.glob("mem_rank*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return sorted(
        rows, key=lambda row: (float(row.get("time", 0.0)), str(row.get("tag", "")))
    )


def _rank_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_rank: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "rank" in row and "allocated_mb" in row:
            by_rank[int(row["rank"])].append(row)
    summary: list[dict[str, Any]] = []
    for rank, rank_rows in sorted(by_rank.items()):
        rank_rows = sorted(rank_rows, key=lambda row: float(row.get("time", 0.0)))
        first = rank_rows[0]
        final = rank_rows[-1]
        steps = {
            int(row["step"])
            for row in rank_rows
            if row.get("step") is not None
            and str(row.get("step")).lstrip("-").isdigit()
        }
        allocated_growth_gb = _mb_to_gb(
            float(final.get("allocated_mb", 0.0))
        ) - _mb_to_gb(float(first.get("allocated_mb", 0.0)))
        slope_mb = _slope_by_step(
            [
                row
                for row in rank_rows
                if row.get("tag") in {"step_start", "step_end"}
                and row.get("step") is not None
            ]
        )
        suspicious_growth = allocated_growth_gb > 0.5 and slope_mb > 0.0
        gpu_utilization_values = _numeric_values(rank_rows, "gpu_utilization_percent")
        gpu_memory_utilization_values = _numeric_values(
            rank_rows, "gpu_memory_utilization_percent"
        )
        gpu_power_values = _numeric_values(rank_rows, "gpu_power_draw_w")
        summary.append(
            {
                "rank": rank,
                "local_rank": final.get("local_rank", ""),
                "hostname": final.get("hostname", ""),
                "device": final.get("device", ""),
                "max_allocated_gb": _round_gb(
                    max(
                        float(row.get("max_allocated_mb", row.get("allocated_mb", 0.0)))
                        for row in rank_rows
                    )
                ),
                "max_reserved_gb": _round_gb(
                    max(
                        float(row.get("max_reserved_mb", row.get("reserved_mb", 0.0)))
                        for row in rank_rows
                    )
                ),
                "final_allocated_gb": _round_gb(float(final.get("allocated_mb", 0.0))),
                "final_reserved_gb": _round_gb(float(final.get("reserved_mb", 0.0))),
                "allocated_growth_gb": round(allocated_growth_gb, 4),
                "mean_gpu_utilization_percent": round(_mean(gpu_utilization_values), 2),
                "max_gpu_utilization_percent": round(
                    max(gpu_utilization_values, default=0.0), 2
                ),
                "mean_gpu_memory_utilization_percent": round(
                    _mean(gpu_memory_utilization_values), 2
                ),
                "max_gpu_memory_utilization_percent": round(
                    max(gpu_memory_utilization_values, default=0.0), 2
                ),
                "mean_gpu_power_draw_w": round(_mean(gpu_power_values), 2),
                "max_gpu_power_draw_w": round(max(gpu_power_values, default=0.0), 2),
                "num_logged_steps": len(steps),
                "suspicious_growth": suspicious_growth,
            }
        )
    return summary


def _region_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("tag", "")).endswith(":summary") and row.get("region"):
            by_region[str(row["region"])].append(row)
    summary: list[dict[str, Any]] = []
    for region, region_rows in sorted(by_region.items()):
        peak_values = [float(row.get("peak_allocated_mb", 0.0)) for row in region_rows]
        delta_values = [
            float(row.get("delta_allocated_mb", 0.0)) for row in region_rows
        ]
        gpu_utilization_values = _numeric_values(region_rows, "gpu_utilization_percent")
        gpu_memory_utilization_values = _numeric_values(
            region_rows, "gpu_memory_utilization_percent"
        )
        summary.append(
            {
                "region": region,
                "max_peak_allocated_gb": _round_gb(max(peak_values, default=0.0)),
                "mean_peak_allocated_gb": _round_gb(_mean(peak_values)),
                "max_delta_allocated_gb": _round_gb(max(delta_values, default=0.0)),
                "mean_delta_allocated_gb": _round_gb(_mean(delta_values)),
                "mean_gpu_utilization_percent": round(_mean(gpu_utilization_values), 2),
                "mean_gpu_memory_utilization_percent": round(
                    _mean(gpu_memory_utilization_values), 2
                ),
                "num_observations": len(region_rows),
            }
        )
    return sorted(
        summary, key=lambda row: float(row["max_peak_allocated_gb"]), reverse=True
    )


def _step_duration_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_rank_step: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("tag") not in {"step_start", "step_end"}:
            continue
        if row.get("step") is None or not _is_finite_number(row.get("time")):
            continue
        try:
            rank = int(row.get("rank", 0))
            step = int(row["step"])
        except (TypeError, ValueError):
            continue
        by_rank_step[(rank, step)][str(row["tag"])] = row

    summary: list[dict[str, Any]] = []
    for (rank, step), pair in sorted(by_rank_step.items()):
        start = pair.get("step_start")
        end = pair.get("step_end")
        if start is None or end is None:
            continue
        duration_s = float(end["time"]) - float(start["time"])
        if duration_s < 0.0:
            continue
        summary.append(
            {
                "rank": rank,
                "step": step,
                "duration_s": round(duration_s, 6),
                "microbatch": start.get("microbatch", ""),
                "microbatch_size": start.get("microbatch_size", ""),
                "gradient_accumulation_step": start.get(
                    "gradient_accumulation_step", ""
                ),
                "seq_len": end.get("seq_len", start.get("seq_len", "")),
                "target_tokens": end.get(
                    "target_tokens", start.get("target_tokens", "")
                ),
                "completion_tokens": end.get(
                    "completion_tokens", start.get("completion_tokens", "")
                ),
                "start_allocated_gb": _round_gb(float(start.get("allocated_mb", 0.0))),
                "end_allocated_gb": _round_gb(float(end.get("allocated_mb", 0.0))),
                "peak_allocated_gb": _round_gb(
                    float(end.get("max_allocated_mb", end.get("allocated_mb", 0.0)))
                ),
            }
        )
    return summary


def _step_region_duration_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for row in rows:
        if not str(row.get("tag", "")).endswith(":summary"):
            continue
        if not row.get("region") or row.get("step") is None:
            continue
        if not _is_finite_number(row.get("duration_s")):
            continue
        try:
            rank = int(row.get("rank", 0))
            step = int(row["step"])
        except (TypeError, ValueError):
            continue
        summary.append(
            {
                "rank": rank,
                "step": step,
                "microbatch": row.get("microbatch", ""),
                "microbatch_size": row.get("microbatch_size", ""),
                "gradient_accumulation_step": row.get("gradient_accumulation_step", ""),
                "region": str(row["region"]),
                "duration_s": round(float(row["duration_s"]), 6),
                "seq_len": row.get("seq_len", ""),
                "target_tokens": row.get("target_tokens", ""),
                "completion_tokens": row.get("completion_tokens", ""),
                "peak_allocated_gb": _round_gb(
                    float(row.get("peak_allocated_mb", row.get("allocated_mb", 0.0)))
                ),
                "delta_allocated_gb": _round_gb(
                    float(row.get("delta_allocated_mb", 0.0))
                ),
            }
        )
    return sorted(
        summary,
        key=lambda item: (
            int(item["rank"]),
            int(item["step"]),
            _sort_number(item.get("microbatch")),
            str(item["region"]),
        ),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _render_html(
    rows: list[dict[str, Any]],
    rank_summary: list[dict[str, Any]],
    region_summary: list[dict[str, Any]],
    step_duration_summary: list[dict[str, Any]],
    step_region_duration_summary: list[dict[str, Any]],
) -> str:
    allocated_points = [
        row
        for row in rows
        if row.get("step") is not None
        and "allocated_mb" in row
        and row.get("tag") in {"step_start", "step_end"}
    ]
    gpu_utilization_points = [
        row for row in rows if _is_finite_number(row.get("gpu_utilization_percent"))
    ]
    gpu_power_points = [
        row for row in rows if _is_finite_number(row.get("gpu_power_draw_w"))
    ]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CUDA Memory Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 32px; color: #202124; }}
    h1, h2 {{ margin: 0 0 12px; }}
    section {{ margin-top: 28px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
    th, td {{ border: 1px solid #d7dce2; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f4f6f8; }}
    svg {{ width: 100%; height: 280px; border: 1px solid #d7dce2; background: #fff; }}
    .interactive-region-duration svg {{ height: 320px; }}
    .plot-controls, .plot-region-controls {{ display: flex; flex-wrap: wrap; gap: 10px 14px; align-items: center; margin: 8px 0; }}
    .plot-controls label, .plot-check {{ display: inline-flex; align-items: center; gap: 6px; }}
    .plot-controls input, .plot-controls select, .plot-controls button {{ font: inherit; padding: 4px 6px; }}
    .plot-controls input[type=number] {{ width: 84px; }}
    .empty {{ color: #667085; }}
  </style>
</head>
<body>
  <h1>CUDA Memory Report</h1>
  <section>
    <h2>Memory Over Time By Rank</h2>
    {_line_plot(allocated_points)}
  </section>
  <section>
    <h2>GPU Utilization Over Time</h2>
    {_metric_time_plot(gpu_utilization_points, value_key="gpu_utilization_percent", label="GPU utilization", suffix="%", max_value=100.0)}
  </section>
  <section>
    <h2>Power Draw Over Time</h2>
    {_metric_time_plot(gpu_power_points, value_key="gpu_power_draw_w", label="Power draw", suffix=" W")}
  </section>
  <section>
    <h2>Step Duration</h2>
    {_step_duration_plot(step_duration_summary)}
  </section>
  <section>
    <h2>Step Peak Memory</h2>
    {_step_peak_memory_plot(step_region_duration_summary)}
  </section>
  <section>
    <h2>Region Duration By Step</h2>
    {_step_region_duration_plot(step_region_duration_summary)}
  </section>
  <section>
    <h2>Peak Memory By Region</h2>
    {_bar_plot(region_summary, label_key="region", value_key="max_peak_allocated_gb", suffix=" GB")}
  </section>
  <section>
    <h2>Memory Delta By Region</h2>
    {_diverging_bar_plot(region_summary, label_key="region", value_key="mean_delta_allocated_gb", suffix=" GB")}
  </section>
  <section>
    <h2>Leak Trend Summary</h2>
    {_table(rank_summary)}
  </section>
  <section>
    <h2>Region Summary</h2>
    {_table(region_summary)}
  </section>
</body>
</html>
"""


def _step_peak_memory_plot(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<p class="empty">No region memory rows found.</p>'
    group_peak: dict[tuple[int, int, str], float] = defaultdict(float)
    group_meta: dict[tuple[int, int, str], dict[str, Any]] = {}
    for row in rows:
        group = (
            int(row.get("rank", 0)),
            int(row.get("step", 0)),
            str(row.get("microbatch", "")),
        )
        group_peak[group] = max(
            group_peak[group],
            float(row.get("peak_allocated_gb", 0.0)),
        )
        group_meta.setdefault(group, row)

    groups = sorted(
        group_peak,
        key=lambda item: (item[0], item[1], _sort_number(item[2])),
    )
    visible_groups = groups[:120]
    max_peak = max(group_peak.values(), default=1.0) or 1.0
    width = 900
    height = 280
    pad_left = 46
    pad_bottom = 34
    pad_top = 18
    plot_width = width - pad_left - 18
    plot_height = height - pad_top - pad_bottom
    bar_gap = 2
    bar_width = max(1.0, (plot_width / len(visible_groups)) - bar_gap)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="step peak memory">'
    ]
    for tick_index in range(5):
        value = max_peak * tick_index / 4
        y = pad_top + plot_height - (value / max_peak) * plot_height
        parts.append(
            f'<line x1="{pad_left - 4}" y1="{y:.1f}" x2="{width - 18}" y2="{y:.1f}" stroke="#eef2f6" />'
            f'<text x="4" y="{y + 4:.1f}" font-size="11">{value:.1f} GB</text>'
        )
    for index, group in enumerate(visible_groups):
        peak = group_peak[group]
        x = pad_left + index * (bar_width + bar_gap)
        bar_height = (peak / max_peak) * plot_height
        y = pad_top + plot_height - bar_height
        meta = group_meta[group]
        title = html.escape(
            f"step {group[1]} microbatch={group[2]} "
            f"peak={peak:.3f} GB seq_len={meta.get('seq_len', '')} "
            f"target_tokens={meta.get('target_tokens', '')}"
        )
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height:.1f}" fill="#0891b2">'
            f"<title>{title}</title></rect>"
        )
    parts.append(
        f'<line x1="{pad_left}" y1="{height - pad_bottom}" x2="{width - 18}" y2="{height - pad_bottom}" stroke="#98a2b3" />'
        f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height - pad_bottom}" stroke="#98a2b3" />'
    )
    parts.append("</svg>")
    peaks = [group_peak[group] for group in groups]
    return (
        "".join(parts)
        + "<p>"
        + f"microbatches {len(groups)} | mean {_mean(peaks):.2f} GB | max {max(peaks):.2f} GB"
        + (" | showing first 120" if len(groups) > 120 else "")
        + "</p>"
    )


def _step_region_duration_plot(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<p class="empty">No region summary durations found.</p>'
    group_order: list[tuple[int, int, str]] = []
    group_values: dict[tuple[int, int, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    group_meta: dict[tuple[int, int, str], dict[str, Any]] = {}
    region_totals: dict[str, float] = defaultdict(float)
    for row in rows:
        group = (
            int(row.get("rank", 0)),
            int(row.get("step", 0)),
            str(row.get("microbatch", "")),
        )
        if group not in group_values:
            group_order.append(group)
            group_meta[group] = row
        region = str(row.get("region", ""))
        duration = float(row.get("duration_s", 0.0))
        group_values[group][region] += duration
        region_totals[region] += duration

    group_order = sorted(
        group_order, key=lambda item: (item[0], item[1], _sort_number(item[2]))
    )
    regions = [
        region
        for region, _ in sorted(
            region_totals.items(), key=lambda item: item[1], reverse=True
        )
    ]
    colors = [
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#4f46e5",
        "#be123c",
        "#0f766e",
        "#a16207",
    ]
    color_by_region = {
        region: colors[index % len(colors)] for index, region in enumerate(regions)
    }
    groups = []
    for group in group_order:
        meta = group_meta[group]
        durations = {
            region: round(group_values[group].get(region, 0.0), 6)
            for region in regions
            if group_values[group].get(region, 0.0) > 0.0
        }
        groups.append(
            {
                "rank": group[0],
                "step": group[1],
                "microbatch": group[2],
                "seq_len": meta.get("seq_len", ""),
                "target_tokens": meta.get("target_tokens", ""),
                "durations": durations,
                "total_s": round(sum(durations.values()), 6),
            }
        )

    payload = html.escape(
        json.dumps(
            {"groups": groups, "regions": regions, "colors": color_by_region},
            separators=(",", ":"),
        )
    )
    region_controls = "".join(
        '<label class="plot-check">'
        f'<input type="checkbox" data-region="{html.escape(region)}" checked> '
        f'<span style="color:{color_by_region[region]}">{html.escape(region)}</span>'
        "</label>"
        for region in regions
    )
    ranks = sorted({group[0] for group in group_order})
    rank_options = "".join(
        f'<option value="{rank}">rank {rank}</option>' for rank in ranks
    )
    initial_window = min(80, max(1, len(groups)))
    max_skip = max(0, len(groups) - 1)
    return f'''
<div class="interactive-region-duration" data-region-duration-json="{payload}">
  <div class="plot-controls">
    <label>Skip first groups
      <input type="number" data-control="skip" min="0" max="{max_skip}" value="0" step="1">
    </label>
    <label>Show groups
      <input type="number" data-control="limit" min="1" max="{len(groups)}" value="{initial_window}" step="1">
    </label>
    <label>Rank
      <select data-control="rank"><option value="all">all ranks</option>{rank_options}</select>
    </label>
    <button type="button" data-action="prev">Prev</button>
    <button type="button" data-action="next">Next</button>
    <button type="button" data-action="reset">Reset</button>
  </div>
  <div class="plot-region-controls">{region_controls}</div>
  <svg viewBox="0 0 900 320" role="img" aria-label="interactive region duration by step"></svg>
  <p data-role="summary"></p>
</div>
<script>
(() => {{
  const root = document.currentScript.previousElementSibling;
  const data = JSON.parse(root.dataset.regionDurationJson);
  const svg = root.querySelector("svg");
  const summary = root.querySelector('[data-role="summary"]');
  const skipInput = root.querySelector('[data-control="skip"]');
  const limitInput = root.querySelector('[data-control="limit"]');
  const rankSelect = root.querySelector('[data-control="rank"]');
  const regionInputs = Array.from(root.querySelectorAll('[data-region]'));
  const width = 900;
  const height = 320;
  const padLeft = 52;
  const padRight = 18;
  const padTop = 18;
  const padBottom = 42;
  const plotWidth = width - padLeft - padRight;
  const plotHeight = height - padTop - padBottom;
  const gap = 2;

  function selectedRegions() {{
    return regionInputs.filter((input) => input.checked).map((input) => input.dataset.region);
  }}

  function filteredGroups() {{
    const rank = rankSelect.value;
    return data.groups.filter((group) => rank === "all" || String(group.rank) === rank);
  }}

  function groupTotal(group, regions) {{
    return regions.reduce((total, region) => total + Number(group.durations[region] || 0), 0);
  }}

  function setSvg(children) {{
    svg.replaceChildren(...children);
  }}

  function el(name, attrs = {{}}, text = "") {{
    const node = document.createElementNS("http://www.w3.org/2000/svg", name);
    for (const [key, value] of Object.entries(attrs)) {{
      node.setAttribute(key, String(value));
    }}
    if (text) node.textContent = text;
    return node;
  }}

  function clampControls(groups) {{
    const maxSkip = Math.max(0, groups.length - 1);
    skipInput.max = String(maxSkip);
    limitInput.max = String(Math.max(1, groups.length));
    let skip = Math.min(Math.max(0, Number(skipInput.value || 0)), maxSkip);
    let limit = Math.min(
      Math.max(1, Number(limitInput.value || 1)),
      Math.max(1, groups.length - skip),
    );
    skipInput.value = String(skip);
    limitInput.value = String(limit);
    return {{ skip, limit }};
  }}

  function render() {{
    const groupsForRank = filteredGroups();
    const regions = selectedRegions();
    const {{ skip, limit }} = clampControls(groupsForRank);
    const visible = groupsForRank.slice(skip, skip + limit);
    const totals = visible.map((group) => groupTotal(group, regions));
    const maxTotal = Math.max(1e-9, ...totals);
    const barWidth = Math.max(1, plotWidth / Math.max(1, visible.length) - gap);
    const nodes = [];

    for (let tick = 0; tick <= 4; tick += 1) {{
      const value = maxTotal * tick / 4;
      const y = padTop + plotHeight - (value / maxTotal) * plotHeight;
      nodes.push(el("line", {{ x1: padLeft - 4, y1: y.toFixed(1), x2: width - padRight, y2: y.toFixed(1), stroke: "#eef2f6" }}));
      nodes.push(el("text", {{ x: 4, y: (y + 4).toFixed(1), "font-size": 11 }}, `${{value.toFixed(1)}}s`));
    }}

    visible.forEach((group, groupIndex) => {{
      const x = padLeft + groupIndex * (barWidth + gap);
      let yCursor = padTop + plotHeight;
      const total = groupTotal(group, regions);
      regions.forEach((region) => {{
        const duration = Number(group.durations[region] || 0);
        if (duration <= 0) return;
        const segmentHeight = Math.max(0.4, duration / maxTotal * plotHeight);
        yCursor -= segmentHeight;
        const rect = el("rect", {{
          x: x.toFixed(1),
          y: yCursor.toFixed(1),
          width: barWidth.toFixed(1),
          height: segmentHeight.toFixed(1),
          fill: data.colors[region] || "#64748b",
        }});
        const title = el("title");
        title.textContent = `step ${{group.step}} microbatch=${{group.microbatch}} region=${{region}} duration=${{duration.toFixed(3)}}s total=${{total.toFixed(3)}}s seq_len=${{group.seq_len}}`;
        rect.appendChild(title);
        nodes.push(rect);
      }});
      if (visible.length <= 80 && (groupIndex % Math.ceil(visible.length / 12 || 1) === 0)) {{
        nodes.push(el("text", {{ x: x.toFixed(1), y: height - 20, "font-size": 10, transform: `rotate(35 ${{x.toFixed(1)}} ${{height - 20}})` }}, `${{group.step}}/${{group.microbatch}}`));
      }}
    }});

    nodes.push(el("line", {{ x1: padLeft, y1: height - padBottom, x2: width - padRight, y2: height - padBottom, stroke: "#98a2b3" }}));
    nodes.push(el("line", {{ x1: padLeft, y1: padTop, x2: padLeft, y2: height - padBottom, stroke: "#98a2b3" }}));
    setSvg(nodes);

    const mean = totals.length ? totals.reduce((a, b) => a + b, 0) / totals.length : 0;
    const first = visible[0];
    const last = visible[visible.length - 1];
    const range = first && last ? ` | visible step/microbatch ${{first.step}}/${{first.microbatch}} to ${{last.step}}/${{last.microbatch}}` : "";
    summary.textContent = `groups ${{groupsForRank.length}} | showing ${{visible.length}} after skipping ${{skip}} | y max ${{maxTotal.toFixed(2)}}s | mean visible total ${{mean.toFixed(2)}}s${{range}}`;
  }}

  root.querySelector('[data-action="prev"]').addEventListener("click", () => {{
    skipInput.value = String(Math.max(0, Number(skipInput.value || 0) - Number(limitInput.value || 1)));
    render();
  }});
  root.querySelector('[data-action="next"]').addEventListener("click", () => {{
    skipInput.value = String(Number(skipInput.value || 0) + Number(limitInput.value || 1));
    render();
  }});
  root.querySelector('[data-action="reset"]').addEventListener("click", () => {{
    skipInput.value = "0";
    limitInput.value = String(Math.min(80, data.groups.length || 1));
    rankSelect.value = "all";
    regionInputs.forEach((input) => {{ input.checked = true; }});
    render();
  }});
  [skipInput, limitInput, rankSelect, ...regionInputs].forEach((input) => input.addEventListener("input", render));
  [skipInput, limitInput].forEach((input) => input.addEventListener("change", render));
  render();
}})();
</script>
'''


def _step_duration_plot(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<p class="empty">No complete step_start/step_end pairs found.</p>'
    width = 900
    height = 280
    pad_left = 46
    pad_bottom = 34
    pad_top = 18
    plot_width = width - pad_left - 18
    plot_height = height - pad_top - pad_bottom
    max_duration = max(float(row.get("duration_s", 0.0)) for row in rows) or 1.0
    visible_rows = rows[:120]
    bar_gap = 2
    bar_width = max(1.0, (plot_width / len(visible_rows)) - bar_gap)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="step duration">'
    ]
    for tick_index in range(5):
        value = max_duration * tick_index / 4
        y = pad_top + plot_height - (value / max_duration) * plot_height
        parts.append(
            f'<line x1="{pad_left - 4}" y1="{y:.1f}" x2="{width - 18}" y2="{y:.1f}" stroke="#eef2f6" />'
            f'<text x="4" y="{y + 4:.1f}" font-size="11">{value:.1f}s</text>'
        )
    for index, row in enumerate(visible_rows):
        duration = float(row.get("duration_s", 0.0))
        x = pad_left + index * (bar_width + bar_gap)
        bar_height = (duration / max_duration) * plot_height
        y = pad_top + plot_height - bar_height
        title = html.escape(
            f"step {row.get('step')} duration={duration:.3f}s "
            f"microbatch={row.get('microbatch', '')} seq_len={row.get('seq_len', '')}"
        )
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="#2563eb">'
            f"<title>{title}</title></rect>"
        )
    parts.append(
        f'<line x1="{pad_left}" y1="{height - pad_bottom}" x2="{width - 18}" y2="{height - pad_bottom}" stroke="#98a2b3" />'
        f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height - pad_bottom}" stroke="#98a2b3" />'
    )
    parts.append("</svg>")
    durations = [float(row.get("duration_s", 0.0)) for row in rows]
    return (
        "".join(parts)
        + "<p>"
        + f"steps {len(rows)} | mean {_mean(durations):.2f}s | max {max(durations):.2f}s"
        + (" | showing first 120" if len(rows) > 120 else "")
        + "</p>"
    )


def _line_plot(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<p class="empty">No step_start or step_end rows found.</p>'
    width = 900
    height = 260
    pad = 32
    steps = [float(row["step"]) for row in rows]
    values = [_mb_to_gb(float(row.get("allocated_mb", 0.0))) for row in rows]
    min_step, max_step = min(steps), max(steps)
    min_value, max_value = 0.0, max(values) if values else 1.0
    if math.isclose(min_step, max_step):
        max_step += 1.0
    if math.isclose(min_value, max_value):
        max_value += 1.0
    by_rank: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_rank[int(row.get("rank", 0))].append(row)
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
    polylines = []
    labels = []
    y_ticks = []
    for tick_index in range(5):
        value = min_value + (max_value - min_value) * tick_index / 4
        y = (
            height
            - pad
            - (value - min_value) / (max_value - min_value) * (height - 2 * pad)
        )
        y_ticks.append(
            f'<line x1="{pad - 4}" y1="{y:.1f}" x2="{width - pad}" y2="{y:.1f}" stroke="#eef2f6" />'
            f'<text x="4" y="{y + 4:.1f}" font-size="11">{value:.1f} GB</text>'
        )
    for index, (rank, rank_rows) in enumerate(sorted(by_rank.items())):
        points = []
        for row in sorted(rank_rows, key=lambda item: float(item["step"])):
            x = pad + (float(row["step"]) - min_step) / (max_step - min_step) * (
                width - 2 * pad
            )
            y = (
                height
                - pad
                - (_mb_to_gb(float(row.get("allocated_mb", 0.0))) - min_value)
                / (max_value - min_value)
                * (height - 2 * pad)
            )
            points.append(f"{x:.1f},{y:.1f}")
        color = colors[index % len(colors)]
        polylines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(points)}" />'
        )
        labels.append(f'<span style="color:{color}">rank {rank}</span>')
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="allocated memory over time">'
        + "".join(y_ticks)
        + f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#98a2b3" />'
        + f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#98a2b3" />'
        + "".join(polylines)
        + "</svg>"
        + "<p>"
        + " | ".join(labels)
        + f" | max {max_value:.2f} GB"
        + "</p>"
    )


def _metric_time_plot(
    rows: list[dict[str, Any]],
    *,
    value_key: str,
    label: str,
    suffix: str,
    max_value: float | None = None,
) -> str:
    if not rows:
        return f'<p class="empty">No {html.escape(label.lower())} rows found.</p>'
    width = 900
    height = 260
    pad = 42
    times = [float(row.get("time", 0.0)) for row in rows]
    values = [
        float(row[value_key]) for row in rows if _is_finite_number(row.get(value_key))
    ]
    min_time, max_time = min(times), max(times)
    min_value = 0.0
    plotted_max = max_value if max_value is not None else max(values, default=1.0)
    if plotted_max <= min_value:
        plotted_max = min_value + 1.0
    if math.isclose(min_time, max_time):
        max_time += 1.0
    by_rank: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_rank[int(row.get("rank", 0))].append(row)
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
    polylines = []
    labels = []
    y_ticks = []
    for tick_index in range(5):
        value = min_value + (plotted_max - min_value) * tick_index / 4
        y = (
            height
            - pad
            - (value - min_value) / (plotted_max - min_value) * (height - 2 * pad)
        )
        y_ticks.append(
            f'<line x1="{pad - 4}" y1="{y:.1f}" x2="{width - pad}" y2="{y:.1f}" stroke="#eef2f6" />'
            f'<text x="4" y="{y + 4:.1f}" font-size="11">{value:.0f}{html.escape(suffix)}</text>'
        )
    for index, (rank, rank_rows) in enumerate(sorted(by_rank.items())):
        points = []
        for row in sorted(rank_rows, key=lambda item: float(item.get("time", 0.0))):
            value = row.get(value_key)
            if not _is_finite_number(value):
                continue
            x = pad + (float(row.get("time", 0.0)) - min_time) / (
                max_time - min_time
            ) * (width - 2 * pad)
            y = (
                height
                - pad
                - (float(value) - min_value)
                / (plotted_max - min_value)
                * (height - 2 * pad)
            )
            points.append(f"{x:.1f},{y:.1f}")
        if not points:
            continue
        color = colors[index % len(colors)]
        polylines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(points)}" />'
        )
        labels.append(f'<span style="color:{color}">rank {rank}</span>')
    observed_max = max(values, default=0.0)
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(label.lower())} over time">'
        + "".join(y_ticks)
        + f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#98a2b3" />'
        + f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#98a2b3" />'
        + "".join(polylines)
        + "</svg>"
        + "<p>"
        + " | ".join(labels)
        + f" | observed max {observed_max:.2f}{html.escape(suffix)}"
        + "</p>"
    )


def _bar_plot(
    rows: list[dict[str, Any]],
    *,
    label_key: str,
    value_key: str,
    suffix: str,
) -> str:
    if not rows:
        return '<p class="empty">No data found.</p>'
    width = 900
    bar_height = 24
    gap = 8
    pad = 140
    height = max(80, len(rows) * (bar_height + gap) + 24)
    max_value = max(float(row.get(value_key, 0.0)) for row in rows) or 1.0
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    for index, row in enumerate(rows[:30]):
        y = 16 + index * (bar_height + gap)
        label = html.escape(str(row.get(label_key, "")))
        value = float(row.get(value_key, 0.0))
        bar_width = (width - pad - 80) * value / max_value
        parts.append(f'<text x="8" y="{y + 17}" font-size="12">{label}</text>')
        parts.append(
            f'<rect x="{pad}" y="{y}" width="{bar_width:.1f}" height="{bar_height}" fill="#2563eb" />'
        )
        parts.append(
            f'<text x="{pad + bar_width + 6:.1f}" y="{y + 17}" font-size="12">{value:.2f}{html.escape(suffix)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _diverging_bar_plot(
    rows: list[dict[str, Any]],
    *,
    label_key: str,
    value_key: str,
    suffix: str,
) -> str:
    if not rows:
        return '<p class="empty">No data found.</p>'
    width = 900
    bar_height = 24
    gap = 8
    label_width = 140
    plot_width = width - label_width - 110
    zero_x = label_width + plot_width / 2
    height = max(80, len(rows) * (bar_height + gap) + 24)
    max_abs_value = max(abs(float(row.get(value_key, 0.0))) for row in rows) or 1.0
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    parts.append(
        f'<line x1="{zero_x:.1f}" y1="8" x2="{zero_x:.1f}" y2="{height - 8}" stroke="#98a2b3" />'
    )
    for index, row in enumerate(rows[:30]):
        y = 16 + index * (bar_height + gap)
        label = html.escape(str(row.get(label_key, "")))
        value = float(row.get(value_key, 0.0))
        bar_width = (plot_width / 2) * abs(value) / max_abs_value
        x = zero_x if value >= 0 else zero_x - bar_width
        color = "#2563eb" if value >= 0 else "#dc2626"
        text_x = zero_x + bar_width + 6 if value >= 0 else zero_x - bar_width - 72
        parts.append(f'<text x="8" y="{y + 17}" font-size="12">{label}</text>')
        parts.append(
            f'<rect x="{x:.1f}" y="{y}" width="{bar_width:.1f}" height="{bar_height}" fill="{color}" />'
        )
        parts.append(
            f'<text x="{text_x:.1f}" y="{y + 17}" font-size="12">{value:.2f}{html.escape(suffix)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<p class="empty">No data found.</p>'
    columns = list(rows[0].keys())
    header = "".join(f"<th>{html.escape(str(column))}</th>" for column in columns)
    body = []
    for row in rows:
        cells = "".join(
            f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns
        )
        body.append(f"<tr>{cells}</tr>")
    return (
        f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


def _slope_by_step(rows: list[dict[str, Any]]) -> float:
    points = [(float(row["step"]), float(row.get("allocated_mb", 0.0))) for row in rows]
    if len(points) < 2:
        return 0.0
    mean_x = _mean([point[0] for point in points])
    mean_y = _mean([point[1] for point in points])
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator == 0.0:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return values


def _is_finite_number(value: Any) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _sort_number(value: Any) -> tuple[int, float | str]:
    if value in {None, ""}:
        return (0, 0.0)
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _mb_to_gb(value: float) -> float:
    return value / 1024.0


def _round_gb(value_mb: float) -> float:
    return round(_mb_to_gb(value_mb), 4)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a CUDA memory profile report."
    )
    parser.add_argument(
        "--input", required=True, help="Directory containing mem_rank*.jsonl files."
    )
    parser.add_argument("--output", help="Path to memory_report.html.")
    args = parser.parse_args()
    output_path = generate_report(args.input, args.output)
    print(output_path)


if __name__ == "__main__":
    main()
