#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["mlflow-skinny>=3.1", "pyyaml>=6"]
# ///
"""The MLflow store kept in Git: the public results repository holds `mlruns/`, MLflow's own
file-store format (one folder per experiment and run, with params, metrics, tags, inputs and
artifacts as files). This script makes that folder usable anywhere:

  relocate  point the store's artifact paths at where the folder is now. MLflow writes absolute
            paths, so run this after checking the branch out, then: mlflow ui --backend-store-uri ./mlruns
  export    copy every experiment and run to a tracking server, e.g. DagsHub. Idempotent: a run
            already exported (tag mlflow_store.source_run_id) is skipped.
  site      write index.html next to the store: a static page (GitHub Pages) listing every
            experiment and run with its metrics, params and links to the artifacts.

  uv run mlflow_store.py relocate mlruns
  MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow MLFLOW_TRACKING_USERNAME=<user> \\
  MLFLOW_TRACKING_PASSWORD=<token> uv run mlflow_store.py export mlruns

(The results repository carries a copy of this file at its root.)
"""

from __future__ import annotations

import argparse
import html
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

SOURCE_TAG = "mlflow_store.source_run_id"
# MLflow keeps the file store in maintenance mode; reading and writing it still works when allowed.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


def relocate(store: Path) -> int:
    """Rewrite artifact_location / artifact_uri in every meta.yaml to this folder's location."""
    store = store.resolve()
    changed = 0
    for meta in store.glob("*/meta.yaml"):
        changed += _rewrite(meta, "artifact_location", meta.parent.as_uri())
    for meta in store.glob("*/*/meta.yaml"):
        changed += _rewrite(meta, "artifact_uri", (meta.parent / "artifacts").as_uri())
        changed += _rewrite(meta, "artifact_location", meta.parent.parent.as_uri())
    print(f"{changed} paths now point into {store}")
    return 0


def _rewrite(meta: Path, key: str, value: str) -> int:
    data = yaml.safe_load(meta.read_text()) or {}
    if key not in data or data[key] == value:
        return 0
    data[key] = value
    meta.write_text(yaml.safe_dump(data, sort_keys=False))
    return 1


def export(store: Path, tracking_uri: str | None) -> int:
    from mlflow import MlflowClient
    from mlflow.entities import ViewType

    if not tracking_uri:
        raise SystemExit("set MLFLOW_TRACKING_URI (and MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD)")
    store = store.resolve()
    source = MlflowClient(store.as_uri())
    target = MlflowClient(tracking_uri)
    exported = skipped = 0
    for experiment in source.search_experiments(view_type=ViewType.ACTIVE_ONLY):
        runs = source.search_runs([experiment.experiment_id], max_results=50000)
        if not runs:
            continue
        found = target.get_experiment_by_name(experiment.name)
        target_id = found.experiment_id if found else target.create_experiment(
            experiment.name, tags={k: v for k, v in experiment.tags.items() if not k.startswith("mlflow.")})
        for run in runs:
            done = target.search_runs([target_id], f"tags.`{SOURCE_TAG}` = '{run.info.run_id}'", max_results=1)
            if done:
                skipped += 1
                continue
            _copy_run(source, target, target_id, run, store / experiment.experiment_id / run.info.run_id)
            exported += 1
            print(f"{experiment.name} / {run.info.run_name}: exported")
    print(f"{exported} runs exported, {skipped} already there, to {tracking_uri}")
    return 0


def _copy_run(source, target, experiment_id: str, run, run_dir: Path) -> None:
    from mlflow.entities import Metric, Param, RunTag

    tags = {**run.data.tags, SOURCE_TAG: run.info.run_id}
    new = target.create_run(experiment_id, start_time=run.info.start_time, tags=tags,
                            run_name=run.info.run_name)
    run_id = new.info.run_id
    params = [Param(k, v) for k, v in run.data.params.items()]
    metrics = [Metric(m.key, m.value, m.timestamp, m.step)
               for key in run.data.metrics for m in source.get_metric_history(run.info.run_id, key)]
    for i in range(0, max(len(params), len(metrics)), 100):
        target.log_batch(run_id, metrics=metrics[i:i + 100], params=params[i:i + 100])
    if run.inputs and run.inputs.dataset_inputs:
        target.log_inputs(run_id, run.inputs.dataset_inputs)
    artifacts = run_dir / "artifacts"  # read from the folder, not the absolute path in meta.yaml
    if artifacts.is_dir() and any(artifacts.iterdir()):
        target.log_artifacts(run_id, str(artifacts))
    target.set_terminated(run_id, run.info.status, end_time=run.info.end_time)

METRICS = ["main_score", "ndcg_at_10", "recall_at_10", "recall_at_100", "mrr_at_10", "precision_at_1"]
PARAMS = ["chunking.strategy", "chunking.chunkSize", "chunking.chunkOverlap", "embedding.model",
          "reranking.strategy", "reranking.model", "eval.queries"]
STYLE = """
:root { color-scheme: light dark; --fg: #1d1d1f; --muted: #6e6e73; --bg: #fff; --line: #d2d2d7; --head: #f5f5f7; }
@media (prefers-color-scheme: dark) { :root { --fg: #f5f5f7; --muted: #a1a1a6; --bg: #161617; --line: #3a3a3c; --head: #232325; } }
body { font: 14px/1.5 system-ui, sans-serif; color: var(--fg); background: var(--bg); margin: 0 auto; padding: 24px 16px; max-width: 1400px; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 17px; margin: 32px 0 8px; } p { color: var(--muted); margin: 0 0 8px; }
.scroll { overflow-x: auto; } table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: left; white-space: nowrap; vertical-align: top; }
th { background: var(--head); font-weight: 600; } td.num { text-align: right; } a { color: #0a84ff; }
"""


def site(store: Path, repo_url: str | None = None) -> int:
    """index.html and .nojekyll in the store's parent folder, so the folder serves as a Pages site
    (the links point into the store, which must be named mlruns/)."""
    from mlflow import MlflowClient
    from mlflow.entities import ViewType

    store = store.resolve()
    out = store.parent
    (out / ".nojekyll").write_text("")  # serve every file as is
    client = MlflowClient(store.as_uri())
    sections, total = [], 0
    experiments = sorted(client.search_experiments(view_type=ViewType.ACTIVE_ONLY), key=lambda e: e.name)
    for experiment in experiments:
        runs = client.search_runs([experiment.experiment_id], max_results=50000, order_by=["attributes.start_time DESC"])
        if not runs:
            continue
        total += len(runs)
        rows = [_row(store, experiment.experiment_id, run) for run in runs]
        head = "".join(f"<th>{html.escape(h)}</th>" for h in ["run", "status", "started", *METRICS, *PARAMS, "artifacts"])
        sections.append(f"<h2>{html.escape(experiment.name)}</h2><p>{len(runs)} runs &middot; experiment id "
                        f"{html.escape(experiment.experiment_id)}</p><div class=scroll><table><tr>{head}</tr>"
                        + "".join(rows) + "</table></div>")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    (out / "index.html").write_text(f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1"><title>Experiment results</title>
<style>{STYLE}</style></head><body><h1>Experiment results</h1>
<p>{total} runs, updated {now}. The data is an MLflow store (file-store format) in
{f'<a href="{html.escape(repo_url)}">this repository</a>' if repo_url else 'this repository'}: clone it, then
<code>uv run mlflow_store.py relocate mlruns</code> to browse it with <code>mlflow ui</code>, or
<code>uv run mlflow_store.py export mlruns</code> to copy it to DagsHub or any MLflow server
(<a href="mlflow_store.py">mlflow_store.py</a>).</p>
{'<p>Background reading: <a href="learning-resources.html">learning resources</a> (AI-generated).</p>'
 if (out / "learning-resources.html").exists() else ""}
{"".join(sections) or "<p>No runs yet.</p>"}</body></html>
""")
    print(f"index.html with {total} runs in {out}")
    return 0


def _row(store: Path, experiment_id: str, run) -> str:
    base = f"mlruns/{experiment_id}/{run.info.run_id}/artifacts"
    artifacts_dir = store / experiment_id / run.info.run_id / "artifacts"
    files = sorted(p.relative_to(artifacts_dir).as_posix() for p in artifacts_dir.rglob("*") if p.is_file()) \
        if artifacts_dir.is_dir() else []
    started = datetime.fromtimestamp(run.info.start_time / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M") \
        if run.info.start_time else ""
    cells = [f"<td>{html.escape(run.info.run_name or run.info.run_id)}</td>", f"<td>{html.escape(run.info.status)}</td>",
             f"<td>{started}</td>"]
    for key in METRICS:
        value = run.data.metrics.get(key)
        cells.append(f"<td class=num>{'' if value is None else f'{value:.4f}'}</td>")
    cells += [f"<td>{html.escape(run.data.params.get(key, ''))}</td>" for key in PARAMS]
    links = " ".join(f'<a href="{html.escape(base + "/" + f)}">{html.escape(f)}</a>' for f in files)
    cells.append(f"<td>{links}</td>")
    return "<tr>" + "".join(cells) + "</tr>"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("relocate", help="fix the artifact paths after a checkout").add_argument("store", type=Path)
    ex = sub.add_parser("export", help="copy every run to a tracking server (MLFLOW_TRACKING_URI)")
    ex.add_argument("store", type=Path)
    ex.add_argument("--to", default=os.environ.get("MLFLOW_TRACKING_URI"), help="tracking URI (env MLFLOW_TRACKING_URI)")
    st = sub.add_parser("site", help="write index.html next to the store (GitHub Pages)")
    st.add_argument("store", type=Path)
    st.add_argument("--repo-url", help="link to the repository holding the store")
    args = parser.parse_args(argv)
    if not args.store.is_dir():
        raise SystemExit(f"{args.store} is not a folder")
    if args.command == "relocate":
        return relocate(args.store)
    if args.command == "site":
        return site(args.store, args.repo_url)
    return export(args.store, args.to)


if __name__ == "__main__":
    sys.exit(main())
