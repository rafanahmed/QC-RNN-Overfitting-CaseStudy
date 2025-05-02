# QuantConnect Walk-Forward RNN Trading Strategy (Case Study: Overfitting in Quant Models)

This project showcases a Recurrent Neural Network (RNN)-based trading strategy built on QuantConnect using walk-forward training methodology. While technically sound in its structure, this strategy **ultimately failed** due to overfitting—a common but critical pitfall in quantitative modeling.

This repository now serves as both a working example and a **cautionary case study** on the importance of out-of-sample testing, feature control, and realistic cost modeling in algorithmic trading.

---

## 🧠 Presentation

A complete breakdown of this strategy's design and its failure points is documented in the accompanying PDF: 

**📎 [Overfitting in ML-Based Quant Trading Strategies – Presentation.pdf](./Overfitting_in_ML_Quant_Strats.pdf)**

The presentation covers:
- What went wrong
- How flashy models can mislead
- Real-world trading frictions (slippage, turnover, etc.)
- Practical takeaways to build more resilient strategies

---

## 📦 Included in This Repository

In addition to the codebase, the following supplementary files are provided for full transparency and analysis:

- `Overfitting_in_ML_Quant_Strats.pdf` — PDF slideshow presented on the strategy’s flaws.
- `Focused Red Shark (1).json` — Full QuantConnect result JSON from the backtest.
- `Focused Red Shark_trades.csv` — CSV file containing the detailed order flow and trade history.
- `Focused Red Shark_logs.txt` — Terminal output/logs from the backtest session.

These files allow you to replicate, inspect, and reflect on the real performance and decision trail of the strategy.

---

## 📂 Project Structure

### `research.ipynb` (Model Training)

Trains a `SimpleRNN` model using an anchored walk-forward training setup.
- Fetches SPY daily price data
- Extracts features like intraday and overnight changes
- Applies `StandardScaler` per training window
- Trains models with early stopping and dropout
- Saves weights and scalers by date to `ObjectStore`

### `main.py` (Trading Algorithm)

Runs the live/backtest algorithm using saved models:
- Loads the closest pre-trained model for the current date
- Applies volatility filtering and hysteresis logic
- Trades long SPY if model predicts positive return
- Tracks exit signal streaks and adjusts holdings

---

## ⚠️ Why the Strategy Failed (Overfitting Breakdown)

Despite solid returns on paper—15% CAR, 68% win rate, Sharpe 0.87—the model suffers from critical flaws:

- **Training on Backtest Data**: The model was trained on the same data it was tested on (2000–2025), which led to inflated performance metrics.
- **Lookahead Bias**: Because training extended into the backtest window, the strategy had "future knowledge" of the market, violating live trading conditions.
- **Too Many Tuned Parameters**: 29 configurable elements made the strategy highly prone to curve-fitting.
- **Low Probabilistic Sharpe Ratio (PSR)**: A PSR of 48% means there's only a coin-flip chance that the Sharpe ratio would be positive in the real world.

This project was featured in a presentation to demonstrate these points in practice.

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

### Step 3: Running the Research Notebook (`research.ipynb`)

* Navigate to the "Research" tab within your project.
* Open `research.ipynb`.
* Ensure the walk-forward configuration (`wf_config`) and model hyperparameters (`model_config`) meet your requirements.
* Run all cells sequentially. This will perform the walk-forward training and save the scalers and weights to your project's `ObjectStore`.

### Step 4: Running the Algorithm (`main.py`)

* Ensure that `main.py` is correctly uploaded.
* Verify that the configuration settings within `Initialize` match the settings used in `research.ipynb`.
* Set your desired backtest start/end dates and initial capital in `main.py`.
* Click the "Build" button to check for errors.
* Click "Backtest" to run the algorithm.

</details>

---

## 🙌 Final Thoughts

This was my first attempt at building a quantitative trading strategy. I’ve kept the project intact to share the learning process transparently. If you're exploring algorithmic finance, let this example remind you: **simple, interpretable, and friction-aware models often outperform complex ones that only succeed in hindsight.**

Feedback and collaborations are always welcome!


