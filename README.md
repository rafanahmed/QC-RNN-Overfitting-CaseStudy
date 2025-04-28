# QuantConnect Walk-Forward RNN Trading Strategy

This project contains two primary files for implementing and researching a trading strategy based on a Recurrent Neural Network (RNN) trained using a walk-forward methodology on the QuantConnect Cloud Platform:

* `main.py` (`main (3).py`): The Python algorithm script for backtesting and live trading, utilizing models trained in the research notebook.
* `research.ipynb` (`research (2).ipynb`): A Jupyter Notebook for training the RNN models using historical data and saving the necessary components for the trading algorithm.

## Strategy Overview

### `research.ipynb` (Model Training)

The `research.ipynb` notebook is responsible for the offline training phase using a walk-forward approach:

* **Data Fetching:** Uses `QuantBook` to fetch historical daily data for SPY.
* **Walk-Forward Training:** Implements an anchored walk-forward methodology. It starts with an initial training period (e.g., 10 years) and retrains the model periodically (e.g., annually) by extending the training window.
* **Feature Engineering:** Calculates features like intraday price changes (`close - open`) and overnight gaps (`open - close.shift(1)`). The target variable is the next day's price change.
* **Preprocessing:** Applies `StandardScaler` to the features. Importantly, the scaler is **re-fitted** on the data corresponding to each specific training window to adapt to changing market statistics.
* **Model Architecture:** Builds a `SimpleRNN` model using TensorFlow/Keras, incorporating `Dropout` and L2 regularization to mitigate overfitting.
* **Training & Validation:** Trains the model using the features and targets for the current window, splitting data for validation and employing `EarlyStopping`.
* **Artifact Extraction & Saving:** After training for each period, it extracts the raw NumPy weights (Wxh, Whh, bh, Why, by) from the Keras model and saves both the weights and the corresponding fitted `StandardScaler` object to QuantConnect's `ObjectStore` using date-stamped keys (e.g., `scaler_YYYYMMDD.pkl`, `weights_YYYYMMDD.pkl`).

### `main.py` (Trading Algorithm)

The `main.py` script (`WalkForwardRnnTradingStrat`) executes the trading strategy using the models prepared by the research notebook:

* **Initialization:** Sets the backtest period, cash, and adds the SPY equity. Configures parameters like lookback period, prediction thresholds, volatility limits, and exit hysteresis.
* **Walk-Forward Model Loading:** Periodically (e.g., monthly check triggered by `Schedule.On`), the script checks the current simulation date and loads the appropriate `StandardScaler` and NumPy weights from the `ObjectStore`. It selects the most recent model trained *before* the current date, ensuring no lookahead bias.
* **Volatility Calculation:** Uses `RateOfChange` and `StandardDeviation` indicators to calculate rolling daily volatility.
* **Feature Calculation:** On each data point (daily bar), it fetches historical data, calculates the 'price_changes' and 'overnight_gaps' features required by the model.
* **Prediction:**
    * Applies the loaded `StandardScaler` to the latest features.
    * Performs a forward pass using the loaded NumPy weights and the scaled features to predict the next day's price change. This forward pass is implemented manually using NumPy operations.
* **Trading Logic:**
    * **Entry:** Goes long SPY (100% portfolio allocation) if the prediction is above a defined threshold AND the calculated daily volatility is below a specified threshold.
    * **Exit:** Liquidates the SPY position if the prediction falls below or equals the threshold for a minimum number of consecutive days (hysteresis).
* **State Management:** Tracks the currently active model's training date and the number of consecutive exit signals.
* **Plotting:** Visualizes the prediction value, daily volatility, and the year of the currently active model's training end date.

This setup creates a pipeline where models are periodically retrained on expanding historical data in `research.ipynb`, and the live/backtesting algorithm (`main.py`) dynamically loads and uses the appropriate pre-trained model based on the current date.

## Setup Instructions

Follow these steps to set up and run this project on QuantConnect:

### Step 1: Create a Project

1.  Navigate to [QuantConnect](https://www.quantconnect.com/) and sign into your account.
2.  Create a new project:
    * Click "Projects" → "Create Project."
    * Provide a descriptive name and select Python as your language.

### Step 2: Upload Files

1.  Open your newly created project.
2.  Click on the "Explorer" tab in the left sidebar.
3.  Select "Upload" and upload both `main (3).py` (rename to `main.py` in QC if desired) and `research (2).ipynb` (rename to `research.ipynb`).

### Step 3: Running the Research Notebook (`research.ipynb`)

* Navigate to the "Research" tab within your project.
* Open `research.ipynb`.
* Ensure the walk-forward configuration (`wf_config`) and model hyperparameters (`model_config`) meet your requirements.
* Run all cells sequentially. This will perform the walk-forward training and save the scalers and weights to your project's `ObjectStore`. **This step must be completed before running `main.py` effectively.**

### Step 4: Running the Algorithm (`main.py`)

* Ensure that `main.py` is correctly uploaded.
* Verify that the configuration settings within `Initialize` (especially `initial_train_start_date` and `retrain_frequency_days`) match the settings used in `research.ipynb`.
* Set your desired backtest start/end dates and initial capital in `main.py`. The start date should be after the end date of the *first* training period in the research notebook.
* Click the "Build" button to check for errors.
* Click "Backtest" to run the algorithm. It will load the models saved by the research notebook as needed during the backtest.

## Requirements and Dependencies

QuantConnect Cloud provides a fully managed environment with pre-installed Python packages including:

* numpy
* pandas
* tensorflow (with Keras)
* scikit-learn (sklearn)
* QuantConnect Libraries (`AlgorithmImports`, `QuantBook`)

No additional external libraries are typically required beyond those provided in the standard QuantConnect environment.

## Support

For any issues, visit the [QuantConnect Documentation](https://www.quantconnect.com/docs/home/home) or utilize the community forums for assistance.

---

## Final Notes

This project demonstrates a walk-forward RNN trading strategy implementation on QuantConnect. The separation into research (training) and main (execution) files allows for systematic model updates and testing. Feedback and suggestions are welcome.
The model’s output and evaluation results are saved in accompanying `.json` and `.csv` files for easy review. Thank you for taking the time to explore this project!
