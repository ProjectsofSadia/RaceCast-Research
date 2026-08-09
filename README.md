# RaceCast

## Cross-Circuit Formula 1 Telemetry Performance Decomposition

RaceCast is an independent motorsport telemetry research project investigating how lap-time differences can be decomposed at the segment level using analytical reconstruction and machine-learning methods, and how well those relationships generalize across circuits.

The study uses publicly accessible Formula 1 qualifying telemetry processed through FastF1 and evaluates performance using **leave-one-circuit-out (LOCO) validation**, where each circuit is held out entirely during model training.

The objective is not race-result prediction. RaceCast focuses on a more fundamental performance-engineering question:

> **Where is lap time gained or lost, what telemetry characteristics are associated with those differences, and how well do those relationships transfer to an unseen circuit?**

---

## Research Paper

### Cross-Circuit Formula 1 Telemetry Performance Decomposition

The complete research artifact documents the experimental design, telemetry-processing pipeline, analytical baseline, machine-learning evaluation, sensitivity analysis, limitations, and results.

**[Read the Research Paper](./paper/RaceCast_Cross_Circuit_F1_Telemetry_Research.pdf)**

Research artifact generated from the RaceCast telemetry analysis pipeline — August 2026.

---

## Dataset

The analysis uses **2024 Formula 1 qualifying telemetry** from five circuits selected to provide substantially different circuit characteristics:

| Circuit | Selected Laps | Drivers | Comparison Pairs |
|---|---:|---:|---:|
| Bahrain | 85 | 20 | 65 |
| Monaco | 143 | 20 | 122 |
| Monza | 87 | 20 | 67 |
| Silverstone | 80 | 15 | 65 |
| Suzuka | 74 | 20 | 53 |

### Final research dataset

- **5 circuits**
- **372 same-driver lap comparisons**
- **2,487 segment-level observations**
- **21 drivers represented**
- Telemetry resampled on a **5 m distance grid**

Raw Formula 1 telemetry is not redistributed in this repository. The analysis pipeline is designed around telemetry obtained through FastF1.

---

## Methodology

RaceCast transforms raw telemetry into segment-level performance observations.

```text
Formula 1 Qualifying Telemetry
              │
              ▼
       Telemetry Cleaning
              │
              ▼
     Distance Resampling
          (5 m grid)
              │
              ▼
     Reference-Lap Selection
              │
              ▼
      Circuit Segmentation
              │
              ▼
 Same-Driver Lap Comparison
              │
              ▼
    Telemetry Feature Engineering
              │
              ▼
     Segment-Time Delta
              │
       ┌──────┴──────┐
       ▼             ▼
Analytical       Machine-Learning
Reconstruction       Models
       │             │
       └──────┬──────┘
              ▼
 Leave-One-Circuit-Out Validation
              │
              ▼
  Sensitivity & Explainability
```

For each circuit, qualifying laps are compared against the fastest valid same-driver reference lap.

Telemetry is aligned by distance rather than timestamp, allowing performance differences to be examined at comparable locations around the circuit.

The target variable is:

**Segment time delta relative to the reference lap.**

---

## Telemetry Features

The analysis derives performance features from multiple telemetry channels.

### Speed and corner behavior
- Entry-speed delta
- Minimum-speed delta
- Mean-speed delta
- Exit-speed delta
- Exit-acceleration proxy

### Braking
- Brake fraction
- Relative braking start
- Braking-zone length

### Throttle
- Mean throttle delta
- Full-throttle fraction
- Relative throttle reapplication

### Powertrain and vehicle state
- Mean RPM delta
- Mean gear delta
- Gear-change delta
- DRS fraction

### Segment context
- Segment length
- Reference segment time

---

## Modeling

Three supervised regression approaches were evaluated:

- **Gradient Boosting**
- **Random Forest**
- **Linear Regression**

Evaluation uses **leave-one-circuit-out validation**.

For every fold:

```text
Train → Four circuits
Test  → One completely held-out circuit
```

The process is repeated until every circuit has served as the unseen test circuit.

This provides a substantially stronger test of cross-circuit transfer than randomly splitting telemetry observations from the same circuits.

---

## Machine-Learning Results

Among the evaluated learned models:

| Model | Mean MAE (s) | Mean RMSE (s) | Mean R² |
|---|---:|---:|---:|
| Gradient Boosting | **0.0563** | **0.0935** | **0.7258** |
| Random Forest | 0.0575 | 0.1012 | 0.6965 |
| Linear Regression | 0.0701 | 0.1045 | 0.6261 |

Gradient Boosting produced the lowest mean held-out-circuit MAE among the evaluated learned models.

Detailed circuit-level results are available in [`results/paper_results.md`](./results/paper_results.md).

---

## A Result That Changed the Research

The strongest machine-learning result was not the strongest overall method.

A non-learned analytical reconstruction based on the relationship between segment distance, reference traversal time, and mean-speed difference achieved approximately:

| Method | Mean MAE |
|---|---:|
| **Analytical reconstruction** | **~0.041 s** |
| Gradient Boosting | ~0.056 s |

The analytical approach therefore outperformed the strongest evaluated learned model.

This finding motivated additional investigation into **target-feature coupling**, particularly because mean speed is mathematically related to segment traversal time.

Rather than treating model accuracy alone as evidence that machine learning had discovered independent performance relationships, RaceCast uses this result to distinguish **telemetry decomposition** from stronger causal or predictive claims.

---

## Feature Sensitivity

Mean-speed delta showed the strongest observed association with segment-time delta.

Among the strongest Spearman associations:

| Feature | Spearman ρ |
|---|---:|
| Mean-speed delta | **-0.823** |
| Mean-gear delta | -0.449 |
| Minimum-speed delta | -0.448 |
| Mean-throttle delta | -0.369 |
| Mean-RPM delta | -0.329 |

Feature-group ablation also showed a substantial deterioration when corner-speed information was removed.

This reinforces an important methodological consideration: some of the strongest predictive telemetry features are closely related to the quantity being reconstructed.

---

## Explainability and Sensitivity Analysis

The research includes:

- Feature-group ablation
- Permutation importance
- SHAP analysis
- Speed-feature sensitivity testing
- Analytical baseline comparison
- Residualized model evaluation
- Circuit-level error analysis

These analyses are used to examine **what information the models depend on**, rather than reporting predictive performance alone.

---

## Key Engineering Takeaway

RaceCast produced a result that is particularly relevant to engineering applications of machine learning:

> **Model complexity must justify itself against a strong domain-informed baseline.**

For segment-level telemetry decomposition, the evaluated machine-learning models captured substantial cross-circuit structure, but a simpler analytical reconstruction produced lower prediction error.

After accounting for the dominant speed–time relationship, the remaining public control and powertrain telemetry provided substantially weaker transferable predictive information.

The result emphasizes the importance of **physics-informed baselines, leakage analysis, sensitivity testing, and careful interpretation of ML performance** in motorsport analytics.

---

## Figures

Research figures include:

- Cross-circuit model comparison
- Feature-group ablation
- Permutation feature importance
- SHAP feature attribution
- Cumulative lap-time delta visualization
- Telemetry speed-trace comparison

Selected publication figures are available in [`figures/`](./figures).

---

## Technology Stack

**Language & Scientific Computing**

`Python` · `NumPy` · `Pandas` · `SciPy`

**Motorsport Telemetry**

`FastF1` · distance-domain telemetry processing · signal smoothing · interpolation · segment analysis

**Machine Learning**

`scikit-learn` · Gradient Boosting · Random Forest · Linear Regression

**Model Evaluation**

Leave-One-Circuit-Out Validation · MAE · RMSE · R² · Spearman Correlation · Feature Ablation

**Explainability**

SHAP · Permutation Importance · Sensitivity Analysis

**Visualization**

Matplotlib · Plotly

---

## Repository Structure

```text
RaceCast-Research/
│
├── paper/
│   └── RaceCast_Cross_Circuit_F1_Telemetry_Research.pdf
│
├── src/
│   └── run_full_analysis.py
│
├── figures/
│   └── publication figures
│
├── results/
│   ├── paper_results.md
│   └── results_summary.json
│
├── docs/
│   ├── METHODOLOGY.md
│   └── REPRODUCIBILITY.md
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Reproducing the Analysis

### 1. Clone the repository

```bash
git clone https://github.com/ProjectsofSadia/RaceCast-Research.git
cd RaceCast-Research
```

### 2. Create a Python environment

Python 3.12 is recommended.

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare telemetry

The analysis expects the processed FastF1 telemetry dataset generated by the RaceCast extraction pipeline.

Raw telemetry and cache files are intentionally excluded from this repository.

See [`docs/REPRODUCIBILITY.md`](./docs/REPRODUCIBILITY.md) for the complete data-generation and execution procedure.

### 5. Run the analysis

```bash
python src/run_full_analysis.py
```

The pipeline generates the research datasets, model evaluation tables, sensitivity analyses, explainability outputs, and publication figures.

---

## Research Scope and Limitations

RaceCast is based on publicly accessible telemetry rather than proprietary team data.

The analysis should therefore be interpreted as an investigation of **predictive associations and performance decomposition**, not as a reconstruction of an F1 team's vehicle-performance model.

Important limitations include:

- Public telemetry has lower channel richness than professional motorsport datasets.
- Speed-derived variables are mathematically coupled with segment traversal time.
- Circuit segmentation is derived from reference telemetry and is not a substitute for a canonical engineering track map.
- Observational telemetry does not establish physical causality.
- Weather, setup, tire state, track evolution, fuel load, and other latent variables cannot be fully controlled using the available data.

These limitations are treated as part of the research rather than hidden from the evaluation.

---

## Research Direction

RaceCast provides a foundation for further work in:

- Canonical circuit segmentation
- Physics-informed lap-time modeling
- Vehicle-dynamics-informed feature engineering
- Tire and degradation modeling
- Uncertainty-aware performance estimation
- Race-strategy optimization
- Simulation and telemetry correlation
- Cross-session and cross-season generalization

---

## Author

**Kazi Sadia Anowar**

Computer Science — Artificial Intelligence  
New York Institute of Technology

Research interests: **Motorsport Engineering · Simulation · Telemetry · Applied Machine Learning · Vehicle Performance Analytics**

**Portfolio:** [kazisadiaanowar.vercel.app](https://kazisadiaanowar.vercel.app/)

---

## Disclaimer

This is an independent research project and is not affiliated with Formula 1, the FIA, Formula One Management, any Formula 1 team, or FastF1.

Formula 1 and related marks are the property of their respective owners.

---

## License

Code in this repository is released under the [MIT License](./LICENSE).

Research data and third-party telemetry remain subject to their respective source terms and licenses.
