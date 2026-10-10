"""Lab 1: Cost & latency dashboard з даних Arize Phoenix.

Джерело даних — тільки Phoenix:
  * спани проєкту (phoenix.client → DataFrame): latency кожного спану, токени, metadata
  * cost кожного trace (GraphQL Trace.costSummary) — Phoenix рахує його сам
    за прайсом моделі, зареєстрованим у phoenix_setup.py

Результат: artifacts/cost_dashboard.html (статичний, без зовнішніх залежностей).

    python dashboard.py [--project acme-support-agent]
"""
import argparse
import html
import json
from datetime import datetime

import httpx
import pandas as pd
from phoenix.client import Client

from agentops import config

KIND_GROUPS = {"LLM": "LLM call", "TOOL": "Tool call (MCP)", "GUARDRAIL": "Guardrail", "CHAIN": "Agent / middleware",
               "AGENT": "Agent / middleware"}
GROUP_ORDER = ["LLM call", "Tool call (MCP)", "Guardrail", "Agent / middleware"]
# Внутрішні спани Guardrails AI (guard/step/call/*.validate) ховаємо — показуємо наші guardrail.*
HIDDEN_SPANS = {"guard", "step", "call"}
# Вузли-обгортки LangGraph дублюють латентність дочірніх спанів (model → ChatOpenAI, tools → MCP tool)
WRAPPER_SPANS = {"model", "tools", "acme-support-agent"}


def _guardrail_failed(attr) -> bool:
    # Phoenix розгортає dotted-атрибути у вкладені dict: guardrail.passed → {"passed": ...}
    return isinstance(attr, dict) and attr.get("passed") is False


def gql(query: str, variables: dict | None = None) -> dict:
    r = httpx.post(f"{config.PHOENIX_URL}/graphql", json={"query": query, "variables": variables or {}}, timeout=30)
    r.raise_for_status()
    body = r.json()
    if body.get("errors"):
        raise RuntimeError(body["errors"])
    return body["data"]


def load(project: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = Client(base_url=config.PHOENIX_URL).spans.get_spans_dataframe(project_identifier=project, limit=100_000)
    df = df[~df["name"].isin(HIDDEN_SPANS) & ~df["name"].str.endswith(".validate")].copy()
    df["latency_ms"] = (df["end_time"] - df["start_time"]).dt.total_seconds() * 1000

    roots = df[df["parent_id"].isna()].copy()
    roots = roots[roots["name"] == "acme-support-agent"].sort_values("start_time")

    proj = next(e["node"] for e in gql("{ projects { edges { node { id name } } } }")["projects"]["edges"]
                if e["node"]["name"] == project)
    totals = gql("query($id: ID!) { node(id: $id) { ... on Project { costSummary { total { cost tokens } } } } }",
                 {"id": proj["id"]})["node"]["costSummary"]["total"]

    runs = []
    for _, r in roots.iterrows():
        tid = r["context.trace_id"]
        t = gql("""query($id: ID!, $tid: ID!) { node(id: $id) { ... on Project {
                    trace(traceId: $tid) { latencyMs costSummary { total { cost tokens } } } } } }""",
                {"id": proj["id"], "tid": tid})["node"]["trace"]
        meta = r.get("attributes.metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        spans = df[df["context.trace_id"] == tid]
        llm = spans[spans["span_kind"] == "LLM"]
        runs.append({
            "trace_id": tid,
            "start": r["start_time"],
            "scenario": meta.get("scenario", "?"),
            "prompt_version": meta.get("prompt_version", "?"),
            "cost": (t["costSummary"]["total"]["cost"] or 0.0),
            "tokens": int(t["costSummary"]["total"]["tokens"] or 0),
            "latency_ms": t["latencyMs"],
            "llm_calls": len(llm),
            "tool_calls": int((spans["span_kind"] == "TOOL").sum()),
            "guardrail_fails": int(sum(_guardrail_failed(a) for a in
                                       spans[spans["name"].str.startswith("guardrail.")].get("attributes.guardrail", []))),
            "status": r["status_code"],
        })
    runs_df = pd.DataFrame(runs)
    df = df[df["context.trace_id"].isin(roots["context.trace_id"])]
    return df, runs_df, totals


# =====================================================================
# SVG helpers (mark specs: bar <= 24px, 4px rounded data-end, hairline grid)
# =====================================================================
def esc(s) -> str:
    return html.escape(str(s), quote=True)


def fmt_usd(v: float) -> str:
    return f"${v:.4f}" if v < 0.01 else f"${v:.3f}"


def fmt_ms(v: float) -> str:
    return f"{v / 1000:.1f}s" if v >= 1000 else f"{v:.0f}ms"


def bar_path(x0, y, w, h, r=4):
    """Горизонтальний бар: квадратний біля baseline, 4px заокруглення на кінці даних."""
    if w <= 0:
        return ""
    r = min(r, w, h / 2)
    return (f"M{x0},{y} H{x0 + w - r} Q{x0 + w},{y} {x0 + w},{y + r} V{y + h - r} "
            f"Q{x0 + w},{y + h} {x0 + w - r},{y + h} H{x0} Z")


def hbar_chart(rows: list[dict], value_key: str, label_key: str, fmt, color_var: str, tip_key: str,
               width=640) -> str:
    """Один ряд (один колір): горизонтальні бари з підписом значення на кінці."""
    label_w, value_w, row_h, bar_h = 190, 70, 30, 18
    plot_w = width - label_w - value_w
    vmax = max([r[value_key] for r in rows] + [1e-12])
    h = row_h * len(rows) + 24
    ticks = [vmax * i / 4 for i in range(5)]
    out = [f'<div class="scroll"><svg viewBox="0 0 {width} {h}" width="100%" role="img" class="chart">']
    for t in ticks:
        x = label_w + plot_w * t / vmax
        out.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{h - 20}" class="grid"/>')
        out.append(f'<text x="{x:.1f}" y="{h - 6}" class="tick" text-anchor="middle">{esc(fmt(t))}</text>')
    out.append(f'<line x1="{label_w}" y1="0" x2="{label_w}" y2="{h - 20}" class="axis"/>')
    for i, r in enumerate(rows):
        y = i * row_h + (row_h - bar_h) / 2
        w = plot_w * r[value_key] / vmax
        out.append(f'<text x="{label_w - 8}" y="{y + bar_h / 2 + 4}" class="lbl" text-anchor="end">'
                   f'{esc(r[label_key])}</text>')
        out.append(f'<g class="hit" data-tip="{esc(r[tip_key])}">'
                   f'<rect x="{label_w}" y="{i * row_h}" width="{plot_w + value_w}" height="{row_h}" fill="transparent"/>'
                   f'<path d="{bar_path(label_w, y, max(w, 2), bar_h)}" fill="var({color_var})"/></g>')
        out.append(f'<text x="{label_w + w + 6}" y="{y + bar_h / 2 + 4}" class="val">{esc(fmt(r[value_key]))}</text>')
    out.append("</svg></div>")
    return "".join(out)


def waterfall(spans: pd.DataFrame, width=640) -> str:
    """Latency кожного спану одного run-у (Gantt). Колір = тип спану (фіксований порядок слотів)."""
    spans = spans.sort_values("start_time")
    t0, t1 = spans["start_time"].min(), spans["end_time"].max()
    total = max((t1 - t0).total_seconds() * 1000, 1)
    label_w, value_w, row_h, bar_h = 210, 64, 24, 14
    plot_w = width - label_w - value_w
    h = row_h * len(spans) + 24
    out = [f'<div class="scroll"><svg viewBox="0 0 {width} {h}" width="100%" role="img" class="chart">']
    for i in range(5):
        x = label_w + plot_w * i / 4
        out.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{h - 20}" class="grid"/>')
        out.append(f'<text x="{x:.1f}" y="{h - 6}" class="tick" text-anchor="middle">{fmt_ms(total * i / 4)}</text>')
    depth = {}
    parents = dict(zip(spans["context.span_id"], spans["parent_id"]))
    for sid in spans["context.span_id"]:
        d, p = 0, parents.get(sid)
        while p in parents:
            d, p = d + 1, parents.get(p)
        depth[sid] = d
    for i, (_, s) in enumerate(spans.iterrows()):
        y = i * row_h + (row_h - bar_h) / 2
        x = label_w + plot_w * (s["start_time"] - t0).total_seconds() * 1000 / total
        w = max(plot_w * s["latency_ms"] / total, 2)
        group = KIND_GROUPS.get(s["span_kind"], "Agent / middleware")
        slot = GROUP_ORDER.index(group) + 1
        indent = min(depth[s["context.span_id"]], 4) * 10
        name = s["name"] if len(s["name"]) <= 30 else s["name"][:29] + "…"
        tip = f'{s["name"]} · {group} · {fmt_ms(s["latency_ms"])}'
        out.append(f'<text x="{4 + indent}" y="{y + bar_h / 2 + 4}" class="lbl">{esc(name)}</text>')
        out.append(f'<g class="hit" data-tip="{esc(tip)}"><rect x="0" y="{i * row_h}" width="{width}" height="{row_h}" '
                   f'fill="transparent"/><path d="{bar_path(x, y, w, bar_h)}" fill="var(--series-{slot})"/></g>')
        out.append(f'<text x="{min(x + w + 6, width - value_w + 8)}" y="{y + bar_h / 2 + 4}" class="val">'
                   f'{fmt_ms(s["latency_ms"])}</text>')
    out.append("</svg></div>")
    return "".join(out)


# =====================================================================
# PAGE
# =====================================================================
CSS = """
.viz-root{color-scheme:light;--page:#f9f9f7;--surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;
--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);
--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--series-4:#eda100;--critical:#d03b3b;--good:#0ca30c}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--page:#0d0d0d;
--surface-1:#1a1a19;--text-primary:#fff;--text-secondary:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--series-4:#c98500}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--page:#0d0d0d;--surface-1:#1a1a19;--text-primary:#fff;
--text-secondary:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--series-4:#c98500}
*{box-sizing:border-box}body{margin:0}
.viz-root{background:var(--page);color:var(--text-primary);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;
min-height:100vh;padding:24px 16px}
.wrap{max-width:1080px;margin:0 auto}h1{font-size:22px;margin:0 0 4px}h2{font-size:15px;margin:0 0 2px}
.sub{color:var(--text-secondary);margin:0 0 20px}.note{color:var(--muted);font-size:12px;margin:2px 0 10px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:16px}
.tile,.card{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.tile .k{color:var(--text-secondary);font-size:12.5px}.tile .v{font-size:26px;font-weight:600;margin-top:2px}
.tile.hero{grid-column:span 2}.tile.hero .v{font-size:48px;line-height:1.05}
@media (max-width:420px){.tile.hero{grid-column:auto}.tile.hero .v{font-size:36px}}.tile .d{color:var(--muted);font-size:12px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:16px;margin-bottom:16px}
.scroll{overflow-x:auto}.chart{display:block;overflow:visible;min-width:560px}.chart .grid{stroke:var(--grid);stroke-width:1}.chart .axis{stroke:var(--axis);stroke-width:1}
.chart text{font:12px system-ui,-apple-system,"Segoe UI",sans-serif}.chart .tick{fill:var(--muted);font-variant-numeric:tabular-nums}
.chart .lbl{fill:var(--text-secondary)}.chart .val{fill:var(--text-primary);font-variant-numeric:tabular-nums}
.chart .hit:hover path{opacity:.8}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12.5px;color:var(--text-secondary);margin:6px 0 8px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:-1px}
select{font:inherit;color:var(--text-primary);background:var(--surface-1);border:1px solid var(--border);border-radius:6px;padding:4px 8px;max-width:100%}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--grid)}
th{color:var(--text-secondary);font-weight:600}td.n{text-align:right;font-variant-numeric:tabular-nums}
.tablewrap{overflow-x:auto}.ok{color:var(--good)}.bad{color:var(--critical)}
#tip{position:fixed;pointer-events:none;background:var(--text-primary);color:var(--surface-1);font-size:12px;padding:5px 8px;
border-radius:6px;display:none;z-index:9;max-width:320px}
"""

JS = """
const tip=document.getElementById('tip');
document.addEventListener('mousemove',e=>{const g=e.target.closest('.hit');
 if(!g){tip.style.display='none';return}tip.textContent=g.dataset.tip;tip.style.display='block';
 tip.style.left=Math.min(e.clientX+12,innerWidth-330)+'px';tip.style.top=(e.clientY+12)+'px'});
const sel=document.getElementById('runsel');
if(sel){const show=()=>document.querySelectorAll('.wf').forEach(d=>d.hidden=d.dataset.trace!==sel.value);
 sel.addEventListener('change',show);show();}
"""


def build(project: str) -> str:
    spans, runs, totals = load(project)
    if runs.empty:
        raise SystemExit(f"No agent runs in Phoenix project '{project}'. Run run_scenarios.py first.")
    runs["label"] = runs.apply(lambda r: f'{r["scenario"]} · {r["start"].strftime("%H:%M")}', axis=1)
    runs["tip"] = runs.apply(lambda r: f'{r["scenario"]}: {fmt_usd(r["cost"])}, {r["tokens"]} tokens, '
                                       f'{r["llm_calls"]} LLM calls, {fmt_ms(r["latency_ms"])}', axis=1)

    n = len(runs)
    avg_cost = runs["cost"].mean()
    max_run = runs.loc[runs["cost"].idxmax()]
    p50, p95 = runs["latency_ms"].quantile(0.5), runs["latency_ms"].quantile(0.95)
    guard_hits = int(runs["guardrail_fails"].sum())

    # Latency по типах/іменах спанів
    sp = spans.copy()
    sp["group"] = sp["span_kind"].map(KIND_GROUPS).fillna("Agent / middleware")
    by_name = (sp[sp["parent_id"].notna()].groupby(["name", "group"])["latency_ms"]
               .agg(count="count", mean="mean", p95=lambda s: s.quantile(0.95), max="max").reset_index()
               .sort_values("mean", ascending=False))
    by_name = by_name[~by_name["name"].isin(WRAPPER_SPANS)]
    top = by_name.head(12).to_dict("records")
    for r in top:
        r["tip"] = f'{r["name"]} ({r["group"]}): avg {fmt_ms(r["mean"])}, p95 {fmt_ms(r["p95"])}, n={r["count"]}'

    # Частка латентності за типом спану (листові спани, без root/model-обгорток)
    leaf_kinds = sp[sp["span_kind"].isin(["LLM", "TOOL"]) | sp["name"].str.startswith("guardrail.")]
    share = leaf_kinds.groupby("group")["latency_ms"].sum()
    share_pct = (share / share.sum() * 100).to_dict()

    cost_rows = runs.to_dict("records")
    by_scn = (runs.groupby("scenario").agg(cost=("cost", "mean"), runs=("cost", "count")).reset_index()
              .sort_values("cost", ascending=False))
    by_scn["tip"] = by_scn.apply(lambda r: f'{r["scenario"]}: avg {fmt_usd(r["cost"])} over {r["runs"]} run(s)', axis=1)

    legend = "".join(f'<span><i style="background:var(--series-{i + 1})"></i>{g}</span>'
                     for i, g in enumerate(GROUP_ORDER))
    options = "".join(f'<option value="{r["trace_id"]}">{esc(r["label"])} — {fmt_ms(r["latency_ms"])}</option>'
                      for r in reversed(cost_rows))
    wfs = "".join(f'<div class="wf" data-trace="{r["trace_id"]}">'
                  f'{waterfall(spans[spans["context.trace_id"] == r["trace_id"]])}</div>' for r in cost_rows)
    rows_html = "".join(
        f'<tr><td>{esc(r["start"].strftime("%Y-%m-%d %H:%M:%S"))}</td><td>{esc(r["scenario"])}</td>'
        f'<td>{esc(r["prompt_version"])}</td><td class="n">{fmt_usd(r["cost"])}</td><td class="n">{r["tokens"]}</td>'
        f'<td class="n">{r["llm_calls"]}</td><td class="n">{r["tool_calls"]}</td><td class="n">{fmt_ms(r["latency_ms"])}</td>'
        f'<td class="n">{r["guardrail_fails"]}</td>'
        f'<td class="{"ok" if r["status"] == "OK" else "bad"}">{esc(r["status"])}</td>'
        f'<td><a href="{config.PHOENIX_URL}/redirects/traces/{r["trace_id"]}" target="_blank">open</a></td></tr>'
        for r in reversed(cost_rows))

    budget_used = max_run["cost"] / config.RUN_BUDGET_USD * 100
    phoenix_total = totals["cost"] or 0.0
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Agent Cost Dashboard</title>
<style>{CSS}</style></head><body><div class="viz-root"><div class="wrap">
<h1>Agent cost &amp; latency dashboard</h1>
<p class="sub">Phoenix project <b>{esc(project)}</b> · model {esc(config.LLM_MODEL)} (local, Ollama) ·
shadow price = <b>{esc(config.PRICE_PROFILE)}</b> · generated {generated}</p>

<div class="tiles">
 <div class="tile hero"><div class="k">Average cost per run</div><div class="v">{fmt_usd(avg_cost)}</div>
  <div class="d">over {n} runs · ≈ ${avg_cost * 1000:.2f} per 1,000 runs</div></div>
 <div class="tile"><div class="k">Most expensive run</div><div class="v">{fmt_usd(max_run["cost"])}</div>
  <div class="d">{esc(max_run["scenario"])} · {budget_used:.1f}% of ${config.RUN_BUDGET_USD:.2f} budget</div></div>
 <div class="tile"><div class="k">Run latency p50 / p95</div><div class="v">{fmt_ms(p50)} / {fmt_ms(p95)}</div>
  <div class="d">end-to-end, root span</div></div>
 <div class="tile"><div class="k">Total cost (Phoenix)</div><div class="v">{fmt_usd(phoenix_total)}</div>
  <div class="d">{int(totals["tokens"] or 0):,} tokens</div></div>
 <div class="tile"><div class="k">Guardrail interventions</div><div class="v">{guard_hits}</div>
  <div class="d">blocked / redacted / sanitized</div></div>
</div>

<div class="grid2">
 <div class="card"><h2>Cost per run ($/run)</h2><p class="note">Each bar is one agent run (trace); cost computed by Phoenix.</p>
 {hbar_chart(cost_rows, "cost", "label", fmt_usd, "--series-1", "tip")}</div>
 <div class="card"><h2>Average $/run by scenario</h2><p class="note">Attacks blocked by the input guardrail cost $0: the LLM is never called.</p>
 {hbar_chart(by_scn.to_dict("records"), "cost", "scenario", fmt_usd, "--series-1", "tip")}</div>
</div>

<div class="grid2">
 <div class="card"><h2>Latency by span (average)</h2>
 <p class="note">All runs. LLM calls take {share_pct.get("LLM call", 0):.1f}% of leaf-span time; tools {share_pct.get("Tool call (MCP)", 0):.2f}%, guardrails {share_pct.get("Guardrail", 0):.2f}%.</p>
 {hbar_chart(top, "mean", "name", fmt_ms, "--series-1", "tip")}</div>
 <div class="card"><h2>Span waterfall for one run</h2>
 <p class="note"><label for="runsel">Run </label><select id="runsel">{options}</select></p>
 <div class="legend">{legend}</div>{wfs}</div>
</div>

<div class="card"><h2>All runs</h2><p class="note">Table view of the charts above; "open" jumps to the trace in Phoenix.</p>
<div class="tablewrap"><table><thead><tr><th>Start</th><th>Scenario</th><th>Prompt</th><th>$ / run</th><th>Tokens</th>
<th>LLM calls</th><th>Tool calls</th><th>Latency</th><th>Guardrail hits</th><th>Status</th><th>Trace</th></tr></thead>
<tbody>{rows_html}</tbody></table></div></div>
</div></div><div id="tip"></div><script>{JS}</script></body></html>"""


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=config.PHOENIX_PROJECT)
    p.add_argument("--out", default=str(config.ARTIFACTS / "cost_dashboard.html"))
    a = p.parse_args()
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(build(a.project))
    print(f"[DASHBOARD] → {a.out}")
