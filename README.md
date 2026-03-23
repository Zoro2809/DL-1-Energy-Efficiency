# 🧠 Deep Learning Project — DL-1
### Energy Efficiency Prediction using Neural Networks from Scratch & TensorFlow/Keras

---



---

## 📌 Project Overview

This Project implements a **2-layer Multilayer Perceptron (MLP)** entirely from scratch using NumPy to predict the **Heating Load** of buildings based on 8 architectural features. It then replicates the same experiments using **TensorFlow/Keras** for validation and comparison.

The primary goal is to develop deep intuition for:
- **Forward & Backward Propagation** (manually coded)
- **Gradient Descent variants** (Batch, SGD, Mini-Batch)
- **Momentum methods** (Classical, Nesterov)
- **Learning Rate Schedules** (Time-Based, Step, Exponential Decay)
- **Adaptive Optimizers** (AdaGrad, RMSProp, Adam, Adamax, Adadelta)
- **Architecture Search** and **Activation Function Selection**

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Name** | ENB2012 — Energy Efficiency Dataset |
| **Source** | UCI Machine Learning Repository |
| **Link** | [UCI ENB2012 Dataset](https://archive.ics.uci.edu/ml/datasets/Energy+efficiency) |
| **File** | `ENB2012_data.xlsx` |
| **Samples** | 768 rows |
| **Features** | 8 input features (X1–X8) |
| **Targets** | Y1 (Heating Load), Y2 (Cooling Load) |
| **Task in this lab** | Regression — predict **Heating Load (Y1)** |

### Feature Description

| Column | Description |
|---|---|
| `Relative_Compactness` | Ratio of building surface to volume |
| `Surface_Area` | Surface area in m² |
| `Wall_Area` | Total wall area |
| `Roof_Area` | Total roof area |
| `Overall_Height` | Building height |
| `Orientation` | Cardinal direction (2–5) |
| `Glazing_Area` | Glazing/window area ratio |
| `Glazing_Area_Distribution` | Glazing distribution variant |
| **`Heating_Load`** | **Target variable (Y1)** |

---

## 🧪 Methodology / Workflow

```
Raw Data (ENB2012_data.xlsx)
        ↓
Feature Renaming & EDA (shape, dtypes, missing values, duplicates)
        ↓
Train/Test Split (80/20, random seed=42)
        ↓
Z-Score Normalization (fit on train, apply to test)
        ↓
═══════════ PHASE A: Manual NumPy MLP ═══════════
        ↓
Architecture: 8 → 5 → 1  (single hidden layer, sigmoid activation)
        ↓
Single Forward Pass → Loss (MSE) → Single Backprop → Weight Update
        ↓
100-epoch Training Loop (Batch GD, lr=0.1)
        ↓
Learning Rate Study (0.001, 0.01, 0.1)
        ↓
Hidden Size Study (1, 2, 3, 5, 10 neurons)
        ↓
Optimizer Comparison (23 optimizers × metrics table)
        ↓
3D Loss Surface Visualization
        ↓
═══════════ PHASE B: TensorFlow / Keras ═══════════
        ↓
Phase 1: Activation + LR Search (relu/sigmoid/tanh × 0.001/0.01/0.1)
        ↓
Phase 2: Architecture Search (1L to 4L funnel networks)
        ↓
Phase 3: GD Variants (Batch / SGD / Mini-Batch)
        ↓
Phase 4: LR Schedules (Time / Step / Exponential)
        ↓
Phase 5: Momentum Methods (Classical / Nesterov)
        ↓
Phase 6: Adaptive Optimizers (AdaGrad, RMSProp, Adam, Adamax, Adadelta)
        ↓
Phase 7: Top-10 Optimizer Bar Chart
        ↓
Phase 8: Manual vs TensorFlow Final Comparison Table
```

---

## 🤖 Algorithms & Techniques Used

### Neural Network Architecture (Manual)
- **Architecture:** `8 → 5 → 1` (single hidden layer)
- **Activation:** Sigmoid (hidden), Linear (output)
- **Loss:** Mean Squared Error (MSE)
- **Initialization:** Random Normal × 0.1, bias = 0

### Neural Network Architecture (Keras Best)
- **Architecture:** `8 → 16 → 8 → 1` (2 hidden layers, funnel design)
- **Activation:** Sigmoid
- **Epochs:** 500, Batch Size: 32 (Mini-Batch)

### Optimizers Studied

| Category | Optimizers |
|---|---|
| **Gradient Descent Variants** | Batch GD, SGD, Mini-Batch GD |
| **LR Decay Schedules** | Time-Based Decay, Step Decay, Exponential Decay |
| **Momentum Methods** | Classical Momentum (β=0.9), Nesterov Accelerated Gradient |
| **Adaptive Optimizers** | AdaGrad, RMSProp, Adam, Adamax, Adadelta |

---

## 📈 Results & Observations

### Phase A — Manual NumPy Results (100 epochs, hidden=10)

| Optimizer | Train Loss (MSE) | Test Loss (MSE) |
|---|---|---|
| Batch GD | 7.8834 | 7.2272 |
| **SGD** | 1.0972 | 1.2202 |
| Mini-Batch | 1.4802 | 1.6511 |
| **SGD + Time Decay** ⭐ | **0.5267** | **0.5466** |
| SGD + Step Decay | 0.7303 | 0.7416 |
| SGD + Momentum | 3.2384 | 3.7466 |
| Mini-Batch + Momentum | 1.0280 | 0.8750 |
| Mini-Batch + Nesterov | 1.3576 | 1.2593 |
| Adam (Mini-Batch) | 99.55 | 71.66 |
| Adadelta (Mini-Batch) | 9.17 | 8.88 |

> 🏆 **Best Manual Optimizer: SGD + Time Decay** — Test MSE = 0.5466, RMSE ≈ 0.74  
> **Why SGD+Time won**: 614 weight updates/epoch + shrinking learning rate = fast & stable convergence.  
> **Why Batch failed**: Only 1 update/epoch — too slow for 100 epochs.  
> **Why Adaptive failed**: Designed for deep networks; need more epochs & smaller LR for this simple dataset.

### Phase B — TensorFlow/Keras Results (500 epochs, 8→16→8→1)

| Phase | Best Configuration | Test Loss |
|---|---|---|
| Activation + LR | Sigmoid, lr=0.01 | ~1.4 |
| Architecture Search | 2L [16→8] | 1.4486 |
| GD Variants | Mini-Batch (bs=32) | 0.2284 |
| LR Schedules | Mini-Batch + Time | < Baseline |
| Momentum | Mini-Batch + Nesterov (lr=0.01) | Low |
| Adaptive | Mini-Batch + Adam | Competitive |

### Graph Explanations

| Plot | What It Shows |
|---|---|
| **Training Loss Curve** | Loss reduces epoch-over-epoch; confirms model is learning |
| **Learning Rate Comparison** | lr=0.1 converges fastest; lr=0.001 too slow in 100 epochs |
| **Hidden Size Study** | More neurons reduce train loss; optimal is ~5–10 for this dataset |
| **Optimizer Comparison (all 23)** | SGD-based variants dominate for simple shallow networks |
| **3D Loss Surface** | Smooth convex bowl confirms gradient descent will find global minimum |
| **Loss Contour Plot** | Elliptical contours indicate feature scale differences |
| **Architecture Search** | Deeper (3L/4L) networks overfit; 2L [16→8] best balances fit vs generalization |
| **Top-10 Bar Chart** | Visual comparison of train vs test loss for best optimizers |

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `DL_1.ipynb` and `ENB2012_data.xlsx`
3. Run all cells from top to bottom

### Option 2: Local (Jupyter/VS Code)
```bash
# 1. Clone or download the project
cd DL-1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter
jupyter notebook DL_1.ipynb
```

> ⚠️ **Note:** The dataset path in the code is `/content/ENB2012_data.xlsx` (Colab default).  
> For local execution, change it to: `df = pd.read_excel("ENB2012_data.xlsx")`

---

## 📦 Requirements

See [`requirements.txt`](requirements.txt) for the full list. Key libraries:

| Library | Purpose |
|---|---|
| `numpy` | Matrix operations, manual neural network |
| `pandas` | Dataset loading and EDA |
| `matplotlib` | All plots and visualizations |
| `openpyxl` | Reading `.xlsx` dataset file |
| `tensorflow` | Keras-based experiments (Phases 1–8) |
| `scikit-learn` | (Optional) metrics comparison |

---

## 📚 Research References

1. **Kingma, D. P., & Ba, J. (2015).** Adam: A Method for Stochastic Optimization. *ICLR 2015.*  
   [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)

2. **Tsanas, A., & Xifara, A. (2012).** Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools. *Energy and Buildings, 49, 560–567.*  
   [https://doi.org/10.1016/j.enbuild.2012.03.003](https://doi.org/10.1016/j.enbuild.2012.03.003)

3. **Ruder, S. (2016).** An Overview of Gradient Descent Optimization Algorithms.  
   [https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)

4. **Nesterov, Y. (1983).** A Method for Solving Convex Programming Problems with Rate of Convergence O(1/k²). *Soviet Mathematics Doklady.*

---

## 🌍 Real-World Applications

| Domain | Application |
|---|---|
| **Green Building Design** | Optimize insulation, glazing, orientation to minimize HVAC load |
| **Smart HVAC Systems** | Predict heating/cooling demand dynamically to reduce energy bills |
| **BIM (Building Information Modelling)** | Integrate ML models into architectural planning tools |
| **Energy Auditing** | Identify inefficient buildings in large housing portfolios |
| **Climate Policy** | Simulate national energy consumption under different building standards |
| **Real Estate** | Predict running costs of properties at design stage |

---

## ⚖️ Comparison with Other Algorithms

| Algorithm | Advantages | Disadvantages |
|---|---|---|
| **MLP (this project)** | Universal approximator, handles non-linearity | Needs tuning; black-box |
| **Linear Regression** | Interpretable, fast | Cannot capture non-linear relationships |
| **Random Forest** | Robust, no scaling needed | Cannot extrapolate, no gradient insight |
| **SVR** | Works well in high-dimensional spaces | Slow on large datasets |
| **XGBoost** | State-of-art for tabular data | No built-in gradient flow analysis |
| **Deep CNN** | Automatic feature extraction | Overkill for tabular data |

### Optimizer Comparison Summary

| Optimizer | Speed | Stability | Best For |
|---|---|---|---|
| **Batch GD** | Slow | High | Small datasets, convex problems |
| **SGD** | Fast | Noisy | Large datasets, simple models |
| **Mini-Batch** | Balanced | Good | General purpose (industry default) |
| **Momentum** | Fast | Medium | Avoiding local minima |
| **Nesterov** | Faster | Good | Better than classical momentum |
| **Adam** | Fast | High | Deep networks, sparse gradients |
| **AdaGrad** | Moderate | Fades | Sparse features / NLP |
| **RMSProp** | Fast | High | RNNs and non-stationary problems |
| **Adadelta** | No LR needed | Medium | Self-tuning applications |

---

## 🗂️ Repository Structure

```
DL-1/
├── DL_1.ipynb              # Main Jupyter/Colab notebook
├── dl_1.py                 # Auto-exported Python script
├── ENB2012_data.xlsx       # Dataset (UCI Energy Efficiency)
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

---

## 🔍 Code Review & Improvement Notes

### ✅ Strengths
- Clean separation of concerns: each optimizer is its own function
- `np.clip(Z, -500, 500)` used in sigmoid for numerical stability
- Reproducibility via `np.random.seed(42)` and `tf.random.set_seed(42)`
- Comprehensive comparison across 23+ optimizer configurations
- Professional architecture selection rationale (funnel principle)

### ⚠️ Minor Issues Found
1. **Dataset path is hardcoded** to `/content/ENB2012_data.xlsx` — needs change for local use
2. **`sgd_time_loss` inconsistency** — at line 1625 it's indexed as `sgd_time_loss[0]` but the function returns a tuple unpacked at line 1157; this could cause an error if run in isolation
3. **`W1[0,0]` and `W1[1,0]`** (lines 2296–2297) are stray expressions with no assignment or print
4. **Adaptive optimizers** (AdaGrad, Adam, etc.) underperform in manual mode because only 100 epochs with suboptimal hyperparameters are used; 500+ epochs would show better results
5. **No RMSE reported for Keras** — only MSE is printed; adding `np.sqrt()` would improve comparability

### 💡 Suggested Improvements
- Add `argparse` or config dict for hyperparameters instead of hardcoded values
- Wrap manual training functions in a unified `Trainer` class
- Add `R² score` alongside MSE for regression evaluation
- Export trained weights to `.npy` files for reproducibility
- Add early stopping callback in Keras experiments
