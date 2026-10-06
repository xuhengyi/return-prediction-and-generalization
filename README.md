# Model Complexity and Generalization in Return Prediction

A reproducible Python study of how model size, regularization, and training time affect financial return prediction. It combines controlled Ridge/Lasso simulations, chronological model selection, and a two-layer neural-network experiment.

The main question is when adding model capacity helps recover a return signal, and when it instead fits estimation noise. A second question is whether a model that generalizes within one regime remains useful after the signal changes.

## Experiments

- **Complexity and shrinkage:** simulate a dense linear signal, vary the number of observed features around and beyond the interpolation threshold, and compare ridgeless regression with fixed Ridge penalties. A penalty calculated from the known data-generating process provides a reference curve.
- **Dense versus sparse estimation:** apply Lasso to the same observations and feature subsets, comparing prediction error, coefficient norm, and return-timing metrics.
- **Chronological prediction:** compare Ridge, Lasso, Elastic Net, and RBF kernel Ridge using expanding training windows. Feature scaling is fitted separately within each training window. A held-out final 20% evaluates the selection procedure; model selection is then repeated on all labeled training data before generating forecasts.
- **Learning time and signal change:** train finite two-layer networks with normalized first-layer weights. Compare validation-based stopping with the final checkpoint, then evaluate the same fitted networks on a rotated latent signal.

The simulated signal and neural-network teacher are known, making estimation error and representation alignment observable. The neural experiment illustrates finite-network behavior; it does not numerically solve dynamical mean-field theory.

## Metrics

The analysis records mean squared error, out-of-sample R² relative to a zero-return forecast, coefficient norms, and return-timing performance. The paper's Sharpe convention is `mean(y * yhat) / sqrt(mean((y * yhat)^2))`; a variance-based Sharpe is also recorded. These measures are not annualized trading returns and do not include transaction costs.

## Quick start

Use Python 3.12 on Linux or WSL. The pinned PyTorch dependency is a CPU build; no GPU is required.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp config.example.json config.json
python -m unittest discover -s tests -v
```

Run the two synthetic experiments without external data:

```bash
python run.py --task simulation
python run.py --task dynamics
```

For a short execution check, use a separate output directory:

```bash
python run.py --task simulation --quick --output outputs/smoke
python run.py --task dynamics --quick --output outputs/smoke
```

`--quick` reduces repetitions or training steps. Full experiments use the fixed grids and seeds in the configuration. The `student` configuration section supplies labels for generated forecasts and report metadata; the example placeholders work for local experiments.

## Prediction data

The prediction pipeline expects three dataset pairs, A/B/C, in a directory supplied with `--data-dir`. Input CSVs are not distributed with this repository.

| File | Columns |
| --- | --- |
| `pairA_train.csv` | `t,feature1,...,featureP,return` |
| `pairA_test_features.csv` | `t,feature1,...,featureP` |

Provide the same filenames for pairs B and C. Each pair may have a different number of features. Timestamps must be strictly increasing integers and all values must be finite. Test feature columns are aligned to the training feature order. Targets are used as provided, without an additional time shift.

```bash
python run.py --task prediction --data-dir private-data
python run.py --task all --data-dir private-data
```

The prediction command writes forecast CSVs with exactly `t,yhat`, fold scores, model rankings, selected-model metadata, and input hashes. The full run also generates plots and a PDF report. Dataset files, local configuration, and generated artifacts are excluded from Git.

## Implementation

| File | Purpose |
| --- | --- |
| `src/core.py` | Metric definitions, stable SVD Ridge solver, and input validation |
| `src/simulation.py` | Paired Ridge and Lasso experiments |
| `src/prediction.py` | Chronological model selection and forecasting |
| `src/dynamics.py` | Projected neural-network updates and checkpoint evaluation |
| `src/figures.py` | Figures from experiment outputs |
| `src/report.py` | Data-driven report generation |
| `tests/` | Metric identities, solver consistency, and CSV schema checks |

`requirements.txt` pins direct dependencies; `requirements-lock.txt` records the reference environment. Fixed random seeds and single-threaded numerical execution improve reproducibility, although different numerical libraries can introduce small floating-point differences.

## Research background

The experiments draw on [The Virtue of Complexity in Return Prediction](https://doi.org/10.1111/jofi.13298), by Kelly, Malamud, and Zhou, and [Dynamical Decoupling of Generalization and Overfitting in Large Two-Layer Networks](https://doi.org/10.52202/085713-0895), by Montanari and Urbani. Bibliographic records are included in `references/`.

Results from synthetic data do not establish real-market profitability. The prediction pipeline reports validation evidence separately from model-search scores, and unlabeled forecast inputs cannot support a claim about hidden-test performance.

## Source archive and license

To export the public source files:

```bash
python scripts/export_public_source.py --list
python scripts/export_public_source.py
```

The exporter writes `dist/project1-public-source.zip` from an explicit allowlist of source files, tests, configuration examples, dependencies, bibliography entries, and documentation. It refuses to overwrite an existing archive; choose a new filename with `--output` when needed.

The source code is licensed under the [MIT License](LICENSE). External datasets and paper PDFs are not included or covered by that license.
