# Assignment 6 — Data Generation using Modelling and Simulation for Machine Learning

**Name:** Akshay Aranveer
**Roll No:** 102303453

---

## Simulation Tool — Cantera

[Cantera](https://cantera.org/) is an open-source toolkit for chemical kinetics, thermodynamics, and transport processes. It is widely used in combustion research, atmospheric chemistry, and electrochemical modelling. Installation is straightforward via pip:

```
pip install cantera
```

---

## Reaction Simulated — Hydrogen-Air Combustion

We simulate **H₂ combustion in air** using Cantera's built-in **H2/O2 mechanism** (`h2o2.yaml`).

### Why Hydrogen?

Hydrogen is one of the most important clean fuels being studied today. Unlike methane, it produces no CO₂ — only water. Its combustion chemistry is also unique: hydrogen has the fastest burning velocity of any conventional fuel, and its ignition behaviour is extremely sensitive to initial conditions, making it a genuinely challenging and interesting problem for machine learning.

### Mechanism Details

| Property | Value |
|---|---|
| Mechanism file | `h2o2.yaml` (built into Cantera) |
| Number of species | 9 |
| Species | H2, H, O, O2, OH, H2O, HO2, H2O2, N2 |
| Number of reactions | 28 elementary reactions |

### Overall Reaction

> **2 H₂ + O₂ → 2 H₂O**

In practice, the reaction passes through many intermediate radical species (OH, H, HO₂, H₂O₂) before reaching the final product. This intermediate chemistry is what makes the ignition threshold non-linear and physically meaningful.

### Reactor Model

We use a **closed, constant-volume, adiabatic batch reactor** (`IdealGasReactor` with `energy='on'`). This model integrates the full system of chemical kinetics ODEs forward in time from the specified initial conditions, capturing the full thermal and chemical evolution of the gas mixture.

---

## Parameter Bounds

| Parameter | Symbol | Description | Lower Bound | Upper Bound |
|---|---|---|---|---|
| Initial Temperature | T | Starting gas temperature | 500 K | 1500 K |
| Initial Pressure | P | Starting gas pressure | 0.5 atm (~50 kPa) | 10 atm (~1 MPa) |
| Equivalence Ratio | φ | Normalised fuel-to-air ratio | 0.3 (lean) | 2.5 (rich) |
| Simulation Time | t_max | Duration reactor is evolved | 0.001 s | 0.5 s |

**Equivalence ratio (φ):**
- φ < 1 → lean mixture (more air than fuel needs)
- φ = 1 → stoichiometric (perfect ratio)
- φ > 1 → rich mixture (excess fuel)

The bounds were chosen to span conditions from well below the auto-ignition threshold (500 K) to well above it (1500 K), and from very lean to very rich mixtures, ensuring the dataset captures a realistic mix of both ignited and non-ignited cases.

---

## Data Generation Methodology

For each of the 1000 simulations:

1. Four parameters (T, P, φ, t_max) were sampled randomly within the bounds above
2. The gas mixture was initialised using `gas.set_equivalence_ratio()` with H₂ as fuel and air (O₂ + N₂ at 3.76:1 ratio) as oxidiser
3. An `IdealGasReactor` was created and evolved forward to `t_max` using `ReactorNet.advance()`
4. The following output quantities were recorded from the reactor state at `t_max`:

| Output Feature | Description |
|---|---|
| T_final | Final temperature of the gas (K) |
| P_final | Final pressure of the gas (Pa) |
| delta_T | Temperature rise = T_final − T_init (K) |
| H2_remain | Mole fraction of H₂ remaining |
| H2O_formed | Mole fraction of H₂O produced |
| OH_conc | Mole fraction of OH radical |
| HO2_conc | Mole fraction of HO₂ radical |
| O2_remain | Mole fraction of O₂ remaining |

5. **Target label:** `ignited = 1` if ΔT > 300 K (significant combustion occurred), else `ignited = 0`

Failed simulations (numerical solver divergence under extreme conditions) were skipped and re-sampled until 1000 valid records were obtained.

---

## ML Models Evaluated

| # | Model | Scaling Applied |
|---|---|---|
| 1 | Logistic Regression | Yes |
| 2 | Decision Tree | No |
| 3 | Random Forest | No |
| 4 | Gradient Boosting | No |
| 5 | AdaBoost | No |
| 6 | SVM (RBF kernel) | Yes |
| 7 | K-Nearest Neighbours | Yes |
| 8 | Naive Bayes | No |
| 9 | MLP Neural Network (128→64→32) | Yes |

An 80/20 stratified train-test split was used. `StandardScaler` was applied for models that are sensitive to feature magnitude.

---

## Evaluation Metrics

| Metric | What it measures |
|---|---|
| Accuracy | Fraction of total correct predictions |
| Precision | Of predicted positives, how many were truly positive |
| Recall | Of actual positives, how many were correctly found |
| F1 Score | Harmonic mean of precision and recall |
| ROC-AUC | Ability to discriminate between classes across all thresholds |

Models are ranked by **F1 Score** as the primary metric since it balances precision and recall, which matters when both false positives and false negatives have physical consequences.

---

## Graphs Included in Notebook

1. **Class distribution** — bar chart and pie chart of ignited vs non-ignited samples
2. **Feature histograms** — distributions of all 8 input/output features
3. **Scatter plots** — T_init vs ΔT, φ vs ΔT, P_init vs ΔT (all coloured by ignition outcome)
4. **Correlation heatmap** — lower-triangle correlation matrix of all features
5. **Grouped bar chart** — all 9 models compared across all 5 metrics
6. **Evaluation heatmap** — colour-coded table of metric scores per model
7. **Confusion matrix** — for the best performing model
8. **Feature importance** — horizontal bar chart from Random Forest
9. **ROC curves** — all 9 models on a single plot with AUC scores

---

## Key Results and Observations

- **Ensemble methods** (Random Forest, Gradient Boosting) achieved the best scores because ignition in hydrogen-air mixtures has a sharp, non-linear threshold that tree-based ensembles handle well
- **Most important features** were `T_final`, `H2O_formed`, and `H2_remain` — these are direct physical indicators of whether combustion occurred
- **Naive Bayes** underperformed because H₂O_formed and H₂_remain are strongly anti-correlated (as one increases the other decreases), violating its feature-independence assumption
- **MLP Neural Network** performed competitively when given scaled features, confirming the ignition boundary is learnable from data alone
- **Logistic Regression** performed reasonably well on linearly separable cases but struggled near the ignition boundary where the decision surface curves sharply
- Hydrogen combustion is more challenging to predict than methane because its much faster chain-branching chemistry means small changes in T or φ can completely flip the ignition outcome

---

## How to Run

1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. Run the first cell — Cantera installs automatically via pip
3. Run all cells in order — the full pipeline (simulation → EDA → ML → plots) runs end to end
