from AlgorithmImports import *
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
# Required for StandardDeviation class name
from QuantConnect.Indicators import RateOfChange, StandardDeviation # Added RateOfChange for clarity

class BasicRnnTradingStrat(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2010, 1, 1)
        self.SetEndDate(2025, 3, 5)
        self.SetCash(100000)
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol

        # --- Configuration ---
        self.config = {
            "lookback": 10,
            "feature_count": 2,
            "threshold": 0.0005,
            "train_end_date": datetime(2010, 1, 1)
        }

        # --- Placeholders ---
        self.scaler = None
        # ... (Wxh, Whh, etc. placeholders) ...
        self.prediction_value = 0
        self.daily_returns = None # Initialize indicator placeholders
        self.volatility = None

        # --- New State Variables ---
        self.exit_signal_days = 0

        # --- Load Pretrained Data ---
        self.LoadPretrainedDataFromObjectStore() # Load first

        # --- Volatility Indicator (CORRECTED - Event Handler Pattern) ---
        # 1. Create the daily returns indicator. self.ROC(...) registers it.
        self.daily_returns = self.ROC(self.spy, 1, Resolution.Daily)

        # 2. Create the StandardDeviation indicator object (give it a name).
        #    It does NOT need explicit registration as it will be updated by the event handler.
        self.volatility = StandardDeviation(f"{self.spy.Value}_ReturnsStdDev", 20)

        # 3. Attach the event handler method (defined below) to the Updated event
        #    of the daily_returns indicator.
        self.daily_returns.Updated += self.OnDailyReturnUpdated

        # --- Warmup Period (Remains 21) ---
        # Warmup required for the ROC(1) + STD(20) chain is 21 bars.
        required_feature_bars = self.config["lookback"] + 1
        required_volatility_bars = 1 + 20 # ROC period + STD period
        self.SetWarmUp(max(required_feature_bars + 5, required_volatility_bars)) # max(16, 21) = 21

        # --- Quit check ---
        if self.scaler is None or self.Wxh is None:
             self.Quit(f"Critical Error at {self.Time}: ML components failed to load.")


    # --- NEW Event Handler Method ---
    def OnDailyReturnUpdated(self, sender, updatedBar):
        """
        Event handler called when self.daily_returns indicator is updated.
        Updates the self.volatility indicator with the new return value.
        """
        # Make sure the volatility indicator is ready before trying to update it
        if self.volatility is not None:
            self.volatility.Update(updatedBar.EndTime, updatedBar.Value)

    # Keep this function (Added stationarity comment)
    def LoadPretrainedDataFromObjectStore(self):
        """Loads the fitted scaler and weights exclusively from ObjectStore."""
        self.Debug(f"Attempting to load scaler and weights from ObjectStore at {self.Time}...")

        scaler_key = "rnn_strategy/scaler.pkl"
        weights_key = "rnn_strategy/weights.pkl"
        load_success = True

        # --- Load Scaler ---
        try:
            scaler_bytes = self.ObjectStore.ReadBytes(scaler_key)
            if scaler_bytes is not None:
                # WARNING: Loading a scaler trained on potentially different data assumes feature statistics are stationary.
                # This approach assumes stationarity; performance may degrade if market regime shifts significantly (concept drift).
                self.scaler = pickle.loads(scaler_bytes)
                self.Debug(f"Successfully loaded scaler from ObjectStore key: '{scaler_key}'")
            else:
                self.Error(f"Scaler key '{scaler_key}' not found in ObjectStore.")
                load_success = False
        except Exception as e:
            self.Error(f"Error loading or deserializing scaler from ObjectStore (key: {scaler_key}): {e}")
            load_success = False

        # --- Load Weights ---
        try:
            weights_bytes = self.ObjectStore.ReadBytes(weights_key)
            if weights_bytes is not None:
                weights = pickle.loads(weights_bytes)
                # Assign weights (ensure keys match exactly what was saved by refactored pretrain.py)
                self.Wxh = np.array(weights["Wxh"])
                self.Whh = np.array(weights["Whh"])
                self.Why = np.array(weights["Why"])
                self.bh  = np.array(weights["bh"])
                self.by  = np.array(weights["by"])
                self.Debug(f"Successfully loaded weights from ObjectStore key: '{weights_key}'")
            else:
                self.Error(f"Weights key '{weights_key}' not found in ObjectStore.")
                load_success = False
        except Exception as e:
            self.Error(f"Error loading or deserializing weights from ObjectStore (key: {weights_key}): {e}")
            load_success = False

        if not load_success:
            self.Error("One or more components failed to load from ObjectStore. Algorithm will Quit in Initialize.")

    # Keep this function as it is needed by OnData
    def forward(self, X):
        """ RNN forward pass - requires weights to be loaded. """
        if self.Wxh is None:
             self.Error("RNN forward pass called before weights were loaded.")
             return np.array([[0.0]])

        # Ensure X has the correct number of features expected by Wxh
        if X.shape[1] != self.Wxh.shape[1]:
            self.Error(f"Feature dimension mismatch in forward pass: Expected {self.Wxh.shape[1]}, got {X.shape[1]}.")
            return np.array([[0.0]])

        h = np.zeros((self.Wxh.shape[0], 1)) # Hidden state shape based on Wxh rows
        for features_at_step_t in X:
            x_t = features_at_step_t.reshape(-1, 1)
            h = np.tanh(np.dot(self.Wxh, x_t) + np.dot(self.Whh, h) + self.bh)
        y = np.dot(self.Why, h) + self.by
        return y


    def OnData(self, data):
        # --- Refactored OnData Logic ---

        # --- Pre-computation checks ---
        if self.IsWarmingUp:
            return

        if not data.ContainsKey(self.spy) or data[self.spy] is None:
             self.Log(f"No data for {self.spy} at {self.Time}")
             return # Wait for data

        # Secondary safety check (should ideally be caught by Initialize Quit)
        if self.scaler is None or self.Wxh is None:
            if not self.IsWarmingUp:
                 self.Error(f"Skipping OnData at {self.Time} - scaler or weights not loaded.")
            return

        # Check if we are past the training period end date defined in config
        if self.Time.date() < self.config["train_end_date"].date():
             return

        # --- History & Feature Calculation (Refactored) ---
        # Request lookback + 1 bars for calculating features like overnight gap
        history = self.History(self.spy, self.config["lookback"] + 1, Resolution.Daily)
        if history.empty or len(history) < self.config["lookback"] + 1:
            # Do not log excessively during normal operation if history is occasionally short
            return

        # Use pandas DataFrame for easier feature calculation
        df_features = pd.DataFrame(index=history.index)
        df_features['open'] = history['open']
        df_features['close'] = history['close']
        # df_features['volume'] = history['volume'] # Volume no longer needed for features
        df_features['price_changes'] = df_features['close'] - df_features['open']
        df_features['overnight_gaps'] = df_features['open'] - df_features['close'].shift(1)
        # df_features['volume_changes'] = df_features['volume'] - df_features['volume'].shift(1) # Removed

        # Updated feature columns list
        feature_cols = ['price_changes', 'overnight_gaps']

        # Get the last 'lookback' rows after calculating shifts (first row has NaN gap)
        raw_features_window = df_features[feature_cols].iloc[1:]

        # Ensure we have the correct number of rows and columns for the model input
        if raw_features_window.shape[0] != self.config["lookback"] or raw_features_window.shape[1] != self.config["feature_count"]:
             self.Log(f"Raw features window shape mismatch at {self.Time}: expected ({self.config['lookback']}, {self.config['feature_count']}), got {raw_features_window.shape}. Check history/feature calculation.")
             return

        # --- Scaling (Refactored: Added comment) ---
        try:
            # WARNING: Using scaler trained on historical data assumes feature statistics are stationary.
            # Performance may degrade if market regime shifts significantly (concept drift).
            features_scaled = self.scaler.transform(raw_features_window)
        except Exception as e:
            self.Error(f"Error scaling features in OnData at {self.Time}: {e}. Raw features head:\n{raw_features_window.head()}")
            return

        # --- RNN Prediction ---
        try:
            prediction = self.forward(features_scaled)
            self.prediction_value = prediction[0][0] # Store scalar value
        except Exception as e:
             self.Error(f"RNN forward pass failed at {self.Time}: {e}")
             self.prediction_value = 0 # Reset prediction on failure
             return # Skip trading if prediction fails

        # --- Plotting ---
        self.Plot("Strategy", "Prediction", float(self.prediction_value))
        if self.volatility.IsReady:
             self.Plot("Strategy", "Volatility", float(self.volatility.Current.Value))

        # --- Trading Logic (Refactored: Volatility Filter, Exit Hysteresis, Logging) ---
        threshold = self.config["threshold"]
        current_volatility = self.volatility.Current.Value if self.volatility.IsReady else -1.0 # Default if not ready
        volatility_threshold = 0.015 # Approx 24% annualized daily std dev

        # --- Exit Signal Update (Refactored) ---
        # Update exit counter based purely on prediction vs threshold, regardless of volatility
        if self.prediction_value <= threshold and self.Portfolio[self.spy].IsLong:
            self.exit_signal_days += 1
            self.Debug(f"{self.Time} - Exit signal detected (Pred: {self.prediction_value:.4f} <= Threshold: {threshold}). Days: {self.exit_signal_days}.")
        elif self.prediction_value > threshold and self.Portfolio[self.spy].IsLong:
            # Reset counter if signal reverses while holding position
            if self.exit_signal_days > 0:
                 self.Debug(f"{self.Time} - Exit signal reset (Pred: {self.prediction_value:.4f} > Threshold: {threshold}).")
            self.exit_signal_days = 0

        # --- Entry Logic (Refactored with Volatility Filter) ---
        if self.prediction_value > threshold and not self.Portfolio[self.spy].IsLong:
            # Only enter if volatility is ready AND within acceptable range
            if self.volatility.IsReady and current_volatility > 0 and current_volatility < volatility_threshold:
                # NOTE: Root cause analysis for cancelled orders often requires reviewing QuantConnect operational logs or contacting support.
                self.Debug(f"{self.Time} - Attempting SetHoldings for {self.spy}. Prediction: {self.prediction_value:.4f}. Volatility: {current_volatility:.4f}. Target: 1.0")
                self.SetHoldings(self.spy, 1.0)
                self.Debug(f"{self.Time} - SetHoldings call completed for {self.spy}.")
            elif not self.volatility.IsReady:
                 self.Debug(f"{self.Time} - Entry condition met (Pred: {self.prediction_value:.4f}) but Volatility indicator not ready. Skipping entry.")
            else:
                 self.Debug(f"{self.Time} - Entry condition met (Pred: {self.prediction_value:.4f}) but Volatility ({current_volatility:.4f}) outside range (0, {volatility_threshold}). Skipping entry.")

        # --- Exit Logic (Refactored with Hysteresis) ---
        # Only liquidate if the exit signal has persisted for 2 days and we are long
        elif self.exit_signal_days >= 2 and self.Portfolio[self.spy].IsLong:
            # NOTE: Root cause analysis for cancelled orders often requires reviewing QuantConnect operational logs or contacting support.
            self.Debug(f"{self.Time} - Attempting Liquidate for {self.spy}. Prediction: {self.prediction_value:.4f}. Exit Signal Days: {self.exit_signal_days}")
            self.Liquidate(self.spy)
            self.Debug(f"{self.Time} - Liquidate call completed for {self.spy}.")
            self.exit_signal_days = 0 # Reset counter after liquidation