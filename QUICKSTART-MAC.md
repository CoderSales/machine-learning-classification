# Mac quickstart

Run commands from the `machine-learning-classification` folder. This setup uses
Python **3.12**. Each repository has its own `.venv`.

## Set up or reuse the environment

```bash
if [ -n "${VIRTUAL_ENV:-}" ]; then deactivate; fi
while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do conda deactivate || break; done
if [ ! -d .venv ]; then
    uv venv --python 3.12 --seed .venv
fi &&
source .venv/bin/activate &&
python -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use a Python 3.12 environment for this project."' &&
python -m pip install -r requirements.txt &&
python -m pip check
```

If `.venv` already exists, it is reused. For later visits, activate it with
`source .venv/bin/activate`. In VS Code, select `.venv/bin/python` as the Python
interpreter and select this environment in the notebook kernel picker.

To open Jupyter:

```bash
python -m notebook
```

## What changed

The previous requirements combined NumPy 2.5 with SciPy 1.10 and Pandas 1.5.
The new resolution uses compatible scientific and Jupyter packages for Python
3.12. Mistune stays at the patched 3.3.3. Windows-only packages have platform
markers, and the obsolete optional `nb-black` formatter is excluded.

## Existing notebooks

The notebook sources are preserved. The older notebooks contain package-install
cells, formatter extension calls, obsolete APIs, and some missing or
Windows-specific data paths. Do not run their install cells: they can overwrite
the environment above with older packages. Skip `%load_ext nb_black` cells;
formatting is not required for the analysis.

The bounded verification uses `DecisionTree_Notebook (1) (3).ipynb` with local
`credit.csv`: imports, encoding, train/test split, one decision tree, metrics,
and a confusion matrix. It excludes the notebook's package-install cell and
large grid search. This does not establish that every notebook runs end to end.
Ensemble notebooks that import XGBoost need that optional dependency separately.

## Update dependencies

Edit the direct requirements in `requirements.in`, then regenerate the resolved
file and test the notebooks you use:

```bash
uv pip compile --upgrade --python-version 3.12 --universal --no-annotate requirements.in --output-file requirements.txt
```

A manual snapshot can be made with `python -m pip freeze > requirements.txt`,
but that overwrites the generated file and does not update `requirements.in`.
Use the compile command above for repository dependency changes.

## Codex in VS Code

Open this repository folder, then use the Command Palette command
**Codex: Open Codex Sidebar**. Sign in with ChatGPT if prompted. Use
**Source Control** to review local file changes before committing.

The VS Code extension and desktop app can edit the same local checkout. Give a
new conversation a short handoff rather than assuming this conversation's
history transfers automatically. Coordinate edits so two agents do not change
the same files at the same time.

Reference: [Official Codex IDE guide](https://learn.chatgpt.com/docs/codex/ide).
