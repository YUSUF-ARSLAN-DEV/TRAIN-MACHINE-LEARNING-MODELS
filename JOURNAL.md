# Journal

## 2026-10-26 — Reproducible runs: seed, config, packaging, pinned dependencies

### What I did
- **Moved hardcoded run values into `config.py`** (ResNet, YOLO; BERT's tokenizer name). Epochs, batch size, learning rate, image size, split ratio and loader settings are now variables I can change in one place.
- **Wrote one `set_seed()`** in `common/seed.py`. It seeds Python's `random`, NumPy and torch, so ResNet's `random_split` and the new final layer's initial weights are the same every run. BERT's split uses NumPy. YOLO has its own `seed=` argument in `train()`, which I set from the same value.
- **Wrote `pyproject.toml`**: the settings file pip reads to build my project as a package. It names the build tool, the project name and version, and which folder (`common`) is importable. `pip install -e .` links it into the venv so `from common.seed import set_seed` works from any folder.
- **Wrote `requirements.in`** (only the libraries I use directly), then ran `pip-compile`. It works out versions that are compatible with each other and writes them all into `requirements.txt`, so a new release of one library can't silently break the others.

### Result
ResNet-18 run twice: **92.8505278755325% both times**, identical to every digit.

### What broke, why, how I fixed it
| Broke | Why | Fix |
|---|---|---|
| `ModuleNotFoundError: common` | The `ok` test ran from the repo root, where Python finds `common` without any install. I had not run `pip install -e .` successfully. | Ran it from the repo root; checked with `pip show` and tested from another folder. |
| Compiled file pinned CPU-only torch | PyPI's Windows torch has no CUDA. The GPU build only exists on PyTorch's own index, which stopped at `2.11.0+cu128`. | Pinned `torch==2.11.0+cu128` and `torchvision==0.26.0+cu128` in `requirements.in`. |
| Edited `requirements.txt` by hand | It's generated, so the next `pip-compile` overwrites it. | Changed `requirements.in` and re-ran the compile. |
| mlflow / protobuf / pyarrow conflict | `onnxruntime` and `datasets` pulled `protobuf 7` and `pyarrow 26`, which MLflow 3 can't use. | Open: see below. |
| Weights saved at the repo root | `path` was a relative path, so it resolves from where I run the command. | Anchored it with `Path(__file__).resolve().parent`. |
| PowerShell errors | `grep` is bash; pasting Python into the terminal runs it as a command. | `Select-String`; run files with `python`. |

### Open
- `requirements.txt` still pins `mlflow==1.26.1` (2022). I need `protobuf<7` and `pyarrow<26` in `requirements.in` so the resolver can pick MLflow 3.

### What I'd explain in an interview
- Seeding fixes the *random* parts (data split, new-layer weights); pinning fixes the *library versions*. Both are needed for the same result twice.
- Why the weights land where you run from: relative paths resolve from the working directory.
- Why `-e .` matters: it links my folder into the venv, so edits take effect without reinstalling.

### Syntax I don't own yet
- An integer index removes a dimension (`boxes[0]` is `(4,)`), a slice keeps it (`boxes[0:1]` is `(1, 4)`).
- `np.stack([...], axis=1)` joins arrays as columns; a plain list of arrays is not one array.
- `np.maximum(a_number, an_array)` compares elementwise and returns an array.
- `random` (Python) and `numpy.random` are different generators with the same name.
- Package names differ from import names: `pillow` is `PIL`, `scikit-learn` is `sklearn`, `pyyaml` is `yaml`.
- Typos that repeat: wrong column index (`keeper[0]` for y), misspelled package names (`fastpi`).
