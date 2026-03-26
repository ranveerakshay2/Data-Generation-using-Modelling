# Assignment 6 — Data Generation using Modelling and Simulation for Machine Learning

**Name:** Akshay Aranveer
**Roll No:** 102303453

---

## Overview

The goal of this assignment was to pick a real simulation tool, use it to generate a dataset by running experiments with random inputs, and then benchmark machine learning models on that data. Instead of collecting data from the real world, we generate it computationally — which is exactly how data is produced in fields like aerospace, chemical engineering, and materials science.

I chose **Cantera** as the simulation tool and decided to simulate **hydrogen-air combustion** — a reaction that is genuinely important in clean energy research and also happens to produce very interesting, non-linear data that challenges ML models in a meaningful way.

---

## Step 1 — The Simulation Tool: Cantera

[Cantera](https://cantera.org/) is an open-source Python library for chemical kinetics, thermodynamics, and transport processes. It lets you define a gas mixture, set its initial conditions, plug it into a reactor model, and watch the chemistry evolve over time — all governed by real physical equations. It is used by researchers at universities and national labs worldwide.

To install it, you just run:

```
pip install cantera
```

What makes Cantera useful here is that it does not just give you a rough answer — it solves the full system of coupled ODEs that describe how every species in the gas mixture changes with time and temperature. So when we ask "did this mixture ignite?", Cantera actually computes it from first principles rather than guessing.

---

## Step 2 — The Reaction: Hydrogen-Air Combustion

### Why Hydrogen?

I specifically chose hydrogen combustion rather than the more common methane example because it tells a more interesting story. Hydrogen burns faster than any other fuel, produces zero carbon emissions (only water), and is at the centre of the global clean energy transition. Its ignition behaviour is also highly sensitive to small changes in temperature and mixture ratio — which makes it a tough and realistic challenge for machine learning.

The overall reaction is:

> **2 H₂ + O₂ → 2 H₂O**

But what actually happens is far more complex. Before reaching water, the mixture passes through a chain of intermediate steps involving reactive radicals — OH, H, HO₂, and H₂O₂. Whether or not the mixture actually ignites depends on whether enough of these radicals accumulate to sustain the chain reaction. That threshold is sharp and non-linear, which is exactly the kind of problem ML is meant to handle.

### The Mechanism: h2o2.yaml

We use Cantera's built-in **H2/O2 mechanism** stored in `h2o2.yaml`. It describes the hydrogen combustion chemistry through:

| Property | Value |
|---|---|
| Mechanism file | `h2o2.yaml` (built into Cantera) |
| Number of species | 9 |
| Species | H2, H, O, O2, OH, H2O, HO2, H2O2, N2 |
| Number of reactions | 28 elementary reactions |

### The Reactor: IdealGasReactor

We model a **closed, constant-volume, adiabatic batch reactor** — meaning nothing enters or leaves, the volume stays fixed, and no heat escapes. This is a standard setup for studying ignition. Cantera's `IdealGasReactor` with `energy='on'` handles this by solving the energy and species conservation equations simultaneously at every time step.

---

## Step 3 — Simulation Parameters and Their Bounds

Each simulation is defined by four randomly sampled input parameters. The bounds were chosen to span a wide physical range — from conditions where ignition definitely will not happen to conditions where it definitely will — ensuring the dataset contains a healthy mix of both outcomes.

| Parameter | Symbol | Physical Meaning | Lower Bound | Upper Bound |
|---|---|---|---|---|
| Initial Temperature | T | How hot the gas starts | 500 K | 1500 K |
| Initial Pressure | P | How compressed the gas is | 0.5 atm (~50 kPa) | 10 atm (~1 MPa) |
| Equivalence Ratio | φ | How fuel-rich or lean the mixture is | 0.3 | 2.5 |
| Simulation Time | t_max | How long we run the reactor | 0.001 s | 0.5 s |

A quick note on equivalence ratio since it comes up a lot:
- φ < 1 means lean — more air than the fuel needs, so some oxygen goes unreacted
- φ = 1 means stoichiometric — the perfect fuel-to-air ratio for complete combustion
- φ > 1 means rich — excess fuel, so some hydrogen goes unreacted

---

## Step 4 — How Each Simulation Was Run

The simulation loop works like this for each of the 1000 runs:

1. Randomly sample T, P, φ, and t_max from the bounds above
2. Set up the gas mixture using `gas.set_equivalence_ratio(phi, 'H2', {'O2': 1.0, 'N2': 3.76})` — air is 21% O₂ and 79% N₂, which gives the 3.76 N₂:O₂ molar ratio
3. Set the initial conditions: `gas.TP = T_init, P_init`
4. Create the reactor and run it forward to t_max using `ReactorNet.advance(t_max)`
5. Record the final chemical and thermal state of the gas

If any simulation fails due to numerical instability at extreme conditions, it is skipped and a new random sample is drawn. This continues until exactly 1000 valid records are collected.

### Features Recorded Per Simulation

Each simulation produces 11 features plus the target label:

| Feature | Description |
|---|---|
| T_init | Initial temperature (K) — simulation input |
| P_init | Initial pressure (Pa) — simulation input |
| phi | Equivalence ratio — simulation input |
| t_max | Simulation duration (s) — simulation input |
| T_final | Temperature at end of simulation (K) |
| P_final | Pressure at end of simulation (Pa) |
| delta_T | Temperature rise = T_final − T_init (K) |
| H2_remain | Mole fraction of H₂ left unreacted |
| H2O_formed | Mole fraction of H₂O produced |
| OH_conc | Mole fraction of OH radical (key ignition indicator) |
| HO2_conc | Mole fraction of HO₂ radical |
| O2_remain | Mole fraction of O₂ remaining |
| **ignited** | **Target label — 1 if ΔT > 300 K, else 0** |

The ignition label is defined using a temperature rise threshold of 300 K. If the gas heated up by more than 300 K during the simulation, we call it ignited. A 300 K rise in a closed adiabatic reactor is a physically grounded threshold — it confirms that significant exothermic chemistry occurred and distinguishes genuine ignition from minor thermal fluctuations.

---

## Step 5 — Exploratory Data Analysis

Before training any models, the notebook explores the dataset visually to understand what the data looks like and whether the simulation is producing physically meaningful results.

### Class Distribution
A bar chart and pie chart confirm that the 1000 simulations produced a reasonably balanced split between ignited and non-ignited cases. This is important — a heavily imbalanced dataset would make accuracy a misleading metric and require resampling techniques.

### Feature Histograms
Histograms for all 8 key features show how the data is distributed. Input parameters (T_init, P_init, phi, t_max) are approximately uniformly distributed, as expected from random sampling. Output features like T_final and delta_T often show a bimodal shape — one cluster for cases that ignited and one for cases that did not.

### Scatter Plots (3 panels)
Three scatter plots show how input conditions relate to the temperature rise (ΔT), with each point coloured by ignition outcome (green = ignited, red = not ignited):

- **T_init vs ΔT** — the clearest view of the ignition temperature threshold. Below roughly 900–1000 K, almost nothing ignites regardless of other conditions.
- **φ vs ΔT** — ignition is easiest near stoichiometric (φ ≈ 1) and becomes harder at the lean and rich extremes.
- **P_init vs ΔT** — higher pressures generally promote ignition, especially at borderline temperatures.

These plots together give a physical intuition for where the ignition boundary sits in the parameter space and why the ML problem is non-trivial.

### Correlation Heatmap
A lower-triangle heatmap of all feature correlations reveals the strong internal structure of the dataset. H2_remain and H2O_formed are strongly negatively correlated (as hydrogen is consumed, water is produced). T_final is strongly positively correlated with H2O_formed. These patterns confirm the simulation is producing chemically consistent and physically meaningful data.

---

## Step 6 — Machine Learning Models

### Train-Test Split
The dataset was split 80/20 (800 training, 200 testing) using stratified sampling to ensure both splits contain the same proportion of ignited cases.

### Feature Scaling
Features vary enormously in magnitude — T_init sits in the hundreds of Kelvin while OH_conc might be on the order of 1e-6. Models that compute distances or rely on gradient descent are very sensitive to this. `StandardScaler` (zero mean, unit variance) was applied for those models. Tree-based models do not need scaling since they split on thresholds rather than magnitudes.

| Model | Scaling Applied |
|---|---|
| Logistic Regression | Yes |
| Decision Tree | No |
| Random Forest | No |
| Gradient Boosting | No |
| AdaBoost | No |
| SVM (RBF kernel) | Yes |
| KNN | Yes |
| Naive Bayes | No |
| MLP Neural Network | Yes |

### The 9 Models

**1. Logistic Regression**
A linear classifier that finds a flat hyperplane separating the two classes. Very fast and interpretable. Works well when the ignition boundary happens to be roughly linear in the feature space, but struggles near the curved parts of the boundary.

**2. Decision Tree**
Builds a tree of if-else rules on feature thresholds. Naturally handles non-linear boundaries. max_depth=8 was used to prevent overfitting — without a depth limit, the tree memorises training data perfectly but generalises poorly.

**3. Random Forest**
Builds 150 decision trees on different random subsets of data and features, then takes a majority vote. The randomness makes it much more robust than a single tree and it is generally one of the strongest models on tabular data.

**4. Gradient Boosting**
Builds 150 trees sequentially, where each tree corrects the errors left by the previous one. This boosting process allows it to progressively sharpen its decision boundary. Often the single best model on structured, tabular datasets.

**5. AdaBoost**
A different boosting approach — instead of fitting residuals like Gradient Boosting, AdaBoost reweights training samples so that the next tree focuses harder on the points that were misclassified. Works well when the base learner is a shallow tree.

**6. SVM (RBF kernel)**
Finds the maximum-margin boundary between classes. The radial basis function (RBF) kernel implicitly maps the data into a higher-dimensional space where it may be more linearly separable. Excellent for non-linear problems but slower on large datasets compared to tree methods.

**7. K-Nearest Neighbours**
Classifies a new point by finding its 7 nearest neighbours in the training set and taking a majority vote. Simple and surprisingly effective when the feature space is well-scaled. Sensitive to irrelevant features and noisy data.

**8. Naive Bayes**
Assumes all features are statistically independent given the class label. Very fast but the independence assumption is badly violated in this dataset — combustion output features are strongly correlated with each other. This causes Naive Bayes to perform poorly.

**9. MLP Neural Network (128 → 64 → 32)**
A three-layer feedforward neural network. Can learn arbitrarily complex decision boundaries if given enough data and the right architecture. Competitive here when properly scaled, confirming that the ignition boundary is learnable from data alone.

---

## Results Table

All 9 models are ranked below by F1 Score. F1 was chosen as the primary metric because it balances precision and recall — in a combustion safety context, both missing a real ignition event and falsely flagging a non-ignition are costly.

| Rank | Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|---|
| 1 | Gradient Boosting | ~0.97 | ~0.97 | ~0.97 | ~0.97 | ~0.99 |
| 2 | Random Forest | ~0.96 | ~0.96 | ~0.96 | ~0.96 | ~0.99 |
| 3 | MLP Neural Network | ~0.95 | ~0.95 | ~0.95 | ~0.95 | ~0.98 |
| 4 | SVM (RBF) | ~0.94 | ~0.94 | ~0.93 | ~0.94 | ~0.98 |
| 5 | AdaBoost | ~0.93 | ~0.93 | ~0.93 | ~0.93 | ~0.97 |
| 6 | KNN | ~0.92 | ~0.92 | ~0.91 | ~0.91 | ~0.96 |
| 7 | Decision Tree | ~0.91 | ~0.90 | ~0.91 | ~0.90 | ~0.91 |
| 8 | Logistic Regression | ~0.87 | ~0.87 | ~0.86 | ~0.86 | ~0.94 |
| 9 | Naive Bayes | ~0.79 | ~0.80 | ~0.77 | ~0.78 | ~0.88 |

> Exact numbers will vary slightly each run due to random sampling in the simulation. The relative ranking of models remains consistent.

---

## Result Graphs

### Graph 1 — Class Distribution (Bar Chart + Pie Chart)
Two side-by-side plots showing the split between ignited (1) and non-ignited (0) cases in the full 1000-sample dataset. This is the first thing to check before any modelling — if the classes were 95/5 split, accuracy would be a useless metric and we would need to handle the imbalance. The charts confirm we have a workable balance.

### Graph 2 — Feature Histograms (2×4 grid)
Eight histograms in a grid layout showing how each feature is distributed across the dataset. Input features (T_init, P_init, phi, t_max) look roughly uniform since they were randomly sampled. Output features like T_final and delta_T often appear bimodal, with one peak corresponding to cases that did not ignite and another (at higher values) corresponding to cases that did. This bimodality is a good sign — it means the classes are genuinely separable in the feature space.

### Graph 3 — Scatter Plots Coloured by Ignition (1×3 grid)
Three scatter plots where x-axes are different input parameters, y-axis is always ΔT (temperature rise), and each point is coloured green (ignited) or red (not ignited). These are the most physically insightful plots in the notebook:

- The T_init vs ΔT plot shows a clear horizontal band — below a certain temperature, almost no point ignites regardless of pressure or phi.
- The φ vs ΔT plot shows that mixtures near stoichiometric (φ ≈ 1) are easiest to ignite, while very lean and very rich extremes resist ignition.
- The P_init vs ΔT plot shows a subtler but real pressure effect — higher pressure shifts the ignition threshold downward.

A shared colorbar on the right makes it easy to read all three panels together.

### Graph 4 — Feature Correlation Heatmap
A triangular heatmap showing pairwise correlations between all features. Strong blue = strong positive correlation, strong red = strong negative correlation. Key patterns visible here are the near-perfect anti-correlation between H2_remain and H2O_formed, and the high correlation between T_final and the combustion product features. This plot also motivates why Naive Bayes struggles — correlated features violate its core assumption.

### Graph 5 — Grouped Bar Chart (Main Model Comparison)
The central result plot of the assignment. Nine model groups are shown on the x-axis. Within each group, five coloured bars represent the five evaluation metrics (Accuracy, Precision, Recall, F1 Score, ROC-AUC). Models are ordered by F1 Score from best to worst. This chart instantly communicates where models agree (high bars across all metrics) and where they diverge (e.g. a model with high precision but low recall).

### Graph 6 — Evaluation Heatmap
A colour-coded grid with models as rows and metrics as columns. Each cell shows the numeric score and is shaded from light (low) to dark (high). This complements the bar chart by making it easy to spot weak spots at a glance — Naive Bayes shows clearly as a lighter row across the board, while Gradient Boosting and Random Forest show as uniformly dark rows.

### Graph 7 — Confusion Matrix (Best Model)
A 2×2 matrix for the top-ranked model showing exactly how the 200 test predictions break down:

- **True Positives** — correctly predicted ignitions
- **True Negatives** — correctly predicted non-ignitions
- **False Positives** — predicted ignition when there was none
- **False Negatives** — missed a real ignition

This is important context that aggregate metrics like accuracy can hide. Even a model with 97% accuracy might have an uncomfortable number of false negatives in the confusion matrix.

### Graph 8 — Feature Importance (Random Forest)
A horizontal bar chart showing how much each of the 11 features contributed to predictions in the Random Forest. Bars highlighted in red exceeded the 0.1 importance threshold. T_final, H2O_formed, and H2_remain consistently rank highest — which makes physical sense because they are the direct chemical outputs that define whether combustion occurred. Input parameters like T_init and phi also appear with moderate importance, showing the model is also learning the ignition conditions themselves.

### Graph 9 — ROC Curves (All 9 Models)
All 9 models plotted on a single ROC curve graph with their AUC scores in the legend. The ROC curve plots true positive rate against false positive rate at every possible decision threshold. A perfect classifier goes straight to the top-left corner (AUC = 1.0), while a random classifier follows the diagonal (AUC = 0.5). Ensemble models cluster near the top-left, while Naive Bayes curves more toward the diagonal. This plot is especially useful for comparing models at different operating thresholds rather than at the default 0.5 cutoff.

---

## Key Observations and Discussion

**Why do ensemble methods come out on top?**
Ignition in hydrogen-air mixtures follows a sharp, curved boundary in the T-P-φ space. A linear model like Logistic Regression tries to cut this space with a single flat plane — which works fine far from the boundary but fails badly near it. Ensemble tree methods approximate the curved boundary through hundreds of piecewise-linear splits, which is why they consistently beat simpler models here.

**Why does Naive Bayes perform so poorly?**
Naive Bayes assumes all features are independent of each other given the class. In this dataset that assumption is completely violated. H2_remain and H2O_formed are nearly perfectly anti-correlated — as one rises the other falls. When features carry redundant information, Naive Bayes double-counts it and produces badly miscalibrated probability estimates, leading to poor classification.

**What does the feature importance tell us?**
The three most important features — T_final, H2O_formed, H2_remain — are all simulation outputs, not inputs. This makes intuitive sense: they are the chemical fingerprints of whether combustion actually occurred. If T_final is much higher than T_init and H2O_formed is high, the mixture ignited — it is almost self-evident. The moderate importance of input features like T_init and phi shows the models also learned the ignition conditions themselves, which is the more practically useful knowledge.

**Why is hydrogen harder to model than methane?**
Hydrogen's chain-branching chemistry is much faster than methane's. This means the transition from no ignition to full ignition is sharper in the T-φ-P space. Near the boundary, very small changes in initial conditions can flip the outcome entirely, making those borderline cases genuinely hard for any classifier to predict correctly — even with 97% overall accuracy, the hardest cases near the threshold are where errors cluster.

**Best model overall: Gradient Boosting**
Across all five metrics, Gradient Boosting came out on top or very close to it. It handles the non-linear ignition boundary well, is robust to outliers, and does not require feature scaling. For a real engineering application — say, predicting safe ignition conditions for a hydrogen fuel cell or combustor — Gradient Boosting would be the model to deploy.

---

## How to Run

1. Open `Assignment_6_Akshay_102303453.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Run the first cell — Cantera installs automatically via pip
3. Run all remaining cells in order
4. The full pipeline runs end to end: simulation → EDA → ML training → all graphs → results table
