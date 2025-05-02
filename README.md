# QuantConnect Walk-Forward RNN Trading Strategy (Case Study: Overfitting in Quant Models)

This project showcases a Recurrent Neural Network (RNN)-based trading strategy built on QuantConnect using walk-forward training methodology. While technically sound in its structure, this strategy **ultimately failed** due to overfitting—a common but critical pitfall in quantitative modeling.

This repository now serves as both a working example and a **cautionary case study** on the importance of out-of-sample testing, feature control, and realistic cost modeling in algorithmic trading.

---

## 🧠 Presentation

A complete breakdown of this strategy's design and its failure points is documented in the accompanying PDF:

**📎 [Balancing ML Models in Quantitative Trading (2).pdf](./Balancing%20ML%20Models%20in%20Quantitative%20Trading%20(2).pdf)**

The presentation covers:
- What went wrong
- How flashy models can mislead
- Real-world trading frictions (slippage, turnover, etc.)
- Practical takeaways to build more resilient strategies

---

## 📦 Included in This Repository

In addition to the codebase, the following supplementary files are provided for full transparency and analysis:

- `Balancing ML Models in Quantitative Trading (2).pdf` — PDF slideshow presented on the strategy’s flaws.
- `Focused Red Shark (1).json` — Full QuantConnect result JSON from the backtest.
- `Focused Red Shark_trades (1).csv` — CSV file containing the detailed order flow and trade history.
- `Focused Red Shark_logs.txt` — Terminal output/logs from the backtest session.
- `main (3).py` — The main Python trading algorithm file.
- `research (2).ipynb` — The Jupyter notebook for model research and training.

These files allow you to replicate, inspect, and reflect on the real performance and decision trail of the strategy.

---

## 📂 Project Structure

### `research (2).ipynb` (Model Training)

Trains a `SimpleRNN` model using an anchored walk-forward training setup.
- Fetches SPY daily price data
- Extracts features like intraday and overnight changes
- Applies `StandardScaler` per training window
- Trains models with early stopping and dropout
- Saves weights and scalers by date to `ObjectStore`

### `main (3).py` (Trading Algorithm)

Runs the live/backtest algorithm using saved models:
- Loads the closest pre-trained model for the current date
- Applies volatility filtering and hysteresis logic
- Trades long SPY if model predicts positive return
- Tracks exit signal streaks and adjusts holdings
- **Note:** Contains a `train_end_date` check inconsistent with the backtest period, requiring adjustment for proper execution.

---

## ⚠️ Why the Strategy Failed (Overfitting Breakdown)

Despite solid returns on paper—15% CAR, 68% win rate, Sharpe 0.87—the model suffers from critical flaws:

- **Training on Backtest Data / Lookahead Bias**: The single biggest issue is that the model was trained on data from 2000-01-01 to 2025-03-05, which entirely encompasses the backtest period of 2010-01-01 to 2025-03-05. This means the model effectively had "future knowledge" during training relative to the backtest, invalidating the results as a measure of true out-of-sample performance.
- **Inflated Metrics**: Consequently, the performance metrics (CAR, Sharpe, Win Rate) are likely significantly overestimated and not representative of real-world potential.
- **Too Many Tuned Parameters**: 29 configurable elements made the strategy highly prone to curve-fitting.
- **Low Probabilistic Sharpe Ratio (PSR)**: A PSR of ~48% suggests only a 50/50 chance that the observed Sharpe ratio is statistically significant and likely positive out-of-sample, further supporting the overfitting concern.
- **Need for Realistic Testing**: Proper validation requires methods like rolling or anchored walk-forward testing where the model is trained only on data available *before* the test period.

This project serves as a practical demonstration of these pitfalls.

---

## ✅ Setup Instructions

See original setup instructions below to reproduce the environment and explore the model files. However, this codebase is now better used as an educational platform rather than a production-ready model.

<details>
<summary>Click to expand original setup steps</summary>

### Step 1: Create a Project

1. Navigate to [QuantConnect](https://www.quantconnect.com/) and sign into your account.
2. Create a new project:
   * Click "Projects" → "Create Project."
   * Provide a descriptive name and select Python as your language.

### Step 2: Upload Files

1. Open your newly created project.
2. Click on the "Explorer" tab in the left sidebar.
3. Select "Upload" and upload both `main (3).py` (rename to `main.py` in QC if desired) and `research (2).ipynb` (rename to `research.ipynb`).

### Step 3: Running the Research Notebook (`research (2).ipynb`)

* Navigate to the "Research" tab within your project.
* Open `research (2).ipynb`.
* Ensure the walk-forward configuration (`wf_config`) and model hyperparameters (`model_config`) meet your requirements.
* Run all cells sequentially. This will perform the walk-forward training and save the scalers and weights to your project's `ObjectStore`.

### Step 4: Running the Algorithm (`main (3).py`)

* Ensure that `main (3).py` is correctly uploaded.
* Verify that the configuration settings within `Initialize` match the settings used in `research (2).ipynb`.
* **Crucially**, adjust the `self.config["train_end_date"]` check in `Initialize` as noted in the analysis to allow trading during the intended backtest period.
* Set your desired backtest start/end dates and initial capital in `main (3).py`.
* Click the "Build" button to check for errors.
* Click "Backtest" to run the algorithm.

</details>

---

## 🙌 Final Thoughts

This was my first attempt at building a quantitative trading strategy. I’ve kept the project intact to share the learning process transparently. If you're exploring algorithmic finance, let this example remind you: **simple, interpretable, and friction-aware models often outperform complex ones that only succeed in hindsight.**

Feedback and collaborations are always welcome!

