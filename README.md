# Experiment results

Benchmark results of [basic-rag-project](https://github.com/ravogdev/basic-rag-project), written by
its Experiments workflow after every run. The page with every experiment and run is this
repository's GitHub Pages site (`index.html`).

- `mlruns/` is an MLflow store in MLflow's file-store format: one folder per experiment and run,
  with params, metrics (MTEB names), tags, dataset inputs and artifacts (`summary.json`,
  `run.trec`, `per_query.csv`, `mteb/`). Experiments are prefixed `test/`.
- `learning-resources.html` is an AI-generated reading list published from the source repository's
  docs by its Docs pages workflow.
- `workflow-runs.jsonl` has one line per workflow run: commit, config and the MLflow run ids it wrote.
- `mlflow_store.py` makes the store usable anywhere:

```bash
uv run mlflow_store.py relocate mlruns     # MLflow stores absolute paths: point them here
MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri ./mlruns

# export everything to a tracking server, e.g. DagsHub (runs already there are skipped)
MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow MLFLOW_TRACKING_USERNAME=<user> \
MLFLOW_TRACKING_PASSWORD=<token> uv run mlflow_store.py export mlruns
```

`mlflow migrate-filestore` turns the store into an MLflow database (`sqlite:///mlflow.db`) without loss.
