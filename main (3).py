#region imports
from AlgorithmImports import *
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
# Required for StandardDeviation class name
from QuantConnect.Indicators import RateOfChange, StandardDeviation
#endregion

class WalkForwardRnnTradingStrat(QCAlgorithm): # Renamed class for clarity

    def Initialize(self):
        '''Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'''

        # --- Backtest Period ---
        # NOTE: The *actual* trading period will be determined by the availability
        # of the first trained model from the walk-forward process in research.ipynb.
        # Set StartDate slightly before the first expected model's applicability.
        # If first model trained on 2000-2009, it's applicable from 2010-01-01.
        self.SetStartDate(2010, 1, 1)
        self.SetEndDate(2025, 3, 5)
        self.SetCash(100000)
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol

        # --- Configuration ---
        # Ref: QuantConnect RNN Strategy Analysis (Updated) - Section 1
        self.config = {
            "lookback": 10,
            "feature_count": 2,
            "threshold": 0.0005,
            "volatility_threshold": 0.015,
            "exit_hysteresis_days": 2, # Days consecutive exit signal needed
            # Walk-Forward specific config
            "retrain_frequency_days": 365, # Must match research notebook
            "initial_train_start_date": datetime(2000, 1, 1) # Must match research notebook
        }

        # --- Placeholders for Active Model ---
        # These will be updated by LoadModelForDate
        # Ref: Addressing Overfitting - Section 2 (Walk-Forward Implementation in main.py)
        self.active_scaler = None
        self.active_Wxh = None
        self.active_Whh = None
        self.active_bh = None
        self.active_Why = None
        self.active_by = None
        self.active_h = None   # Hidden state (re-initialized before each forward pass)
        self.model_loaded_for_date = None # Tracks the date the current model applies FROM

        # --- Other Placeholders & State ---
        self.prediction_value = 0
        self.daily_returns = None
        self.volatility = None
        self.exit_signal_days = 0

        # --- Volatility Indicator Setup ---
        # Ref: QuantConnect RNN Strategy Analysis (Updated) - Section 1 (Indicator Chain)
        self.daily_returns = self.ROC(self.spy, 1, Resolution.Daily)
        self.volatility = StandardDeviation(20)
        self.daily_returns.Updated += self.OnDailyReturnUpdated

        # --- Warmup ---
        # Needs to cover feature lookback and volatility indicator period
        warmup_period = max(self.config["lookback"] + 1, 21)
        self.SetWarmUp(warmup_period)

        # --- Scheduled Model Update ---
        # Schedule event to check for and load new models periodically.
        # Run slightly after market open to ensure the date is current.
        # Ref: Addressing Overfitting - Section 2 (Walk-Forward loading)
        self.Schedule.On(self.DateRules.MonthStart(self.spy), # Check monthly for simplicity
                         self.TimeRules.AfterMarketOpen(self.spy, 5),
                         self.UpdateModel)

        # --- Initial Model Load Attempt ---
        # Try loading the first model applicable at the start date during initialization
        self.UpdateModel() # Call immediately to load the initial model

        # --- Plotting ---
        prediction_chart = Chart('Prediction')
        prediction_chart.AddSeries(Series('Value', SeriesType.Line, 0))
        self.AddChart(prediction_chart)

        volatility_chart = Chart('Volatility')
        volatility_chart.AddSeries(Series('Daily Volatility', SeriesType.Line, 1))
        self.AddChart(volatility_chart)

        # Chart for Model Date
        model_chart = Chart('Model Info')
        model_chart.AddSeries(Series('Model End Date Year', SeriesType.Line, 2)) # Plot year for simplicity
        self.AddChart(model_chart)


    def UpdateModel(self):
        """
        Checks the current date and loads the appropriate walk-forward model
        (scaler and weights) trained *before* this date.
        Ref: Addressing Overfitting - Section 2 (Walk-Forward loading)
        """
        current_date = self.Time.date()
        self.Debug(f"UpdateModel called for date: {current_date.strftime('%Y-%m-%d')}")

        # Determine the key for the model trained *before* the current period started.
        # The key uses the END date of the training period. We need the most recent
        # model trained *before* today.

        # Calculate potential training end dates based on the schedule in research.ipynb
        potential_training_end_dates = []
        start_date = self.config["initial_train_start_date"]
        # Calculate the end date of the initial training period (approx 10 years)
        current_potential_end = start_date + timedelta(days=10 * 365)
        while current_potential_end.date() < current_date: # Only consider models trained before today
             potential_training_end_dates.append(current_potential_end.date())
             current_potential_end += timedelta(days=self.config["retrain_frequency_days"])
             # Ensure we don't look beyond the algorithm's end date if needed
             # (though the loop condition already handles this)

        if not potential_training_end_dates:
            self.Log(f"No potential training end dates found before {current_date}. Cannot load model yet.")
            return

        # Find the latest training end date that is BEFORE the current simulation date
        try:
            target_training_end_date = max(d for d in potential_training_end_dates if d < current_date)
        except ValueError:
            # This can happen if potential_training_end_dates is empty or all dates are >= current_date
            self.Log(f"Could not find a suitable past training end date before {current_date}. Keeping existing model (if any).")
            return


        if target_training_end_date == self.model_loaded_for_date:
            # self.Debug(f"Model for period ending {target_training_end_date} already loaded. No update needed.")
            return # Already have the correct model loaded

        self.Log(f"Attempting to load model trained up to {target_training_end_date.strftime('%Y-%m-%d')} for current date {current_date.strftime('%Y-%m-%d')}")

        # Construct keys based on the target training end date
        period_suffix = target_training_end_date.strftime('%Y%m%d')
        scaler_key = f"rnn_strategy/scaler_{period_suffix}.pkl"
        weights_key = f"rnn_strategy/weights_{period_suffix}.pkl"

        try:
            # Load Scaler
            scaler_bytes = self.ObjectStore.ReadBytes(scaler_key)
            if not scaler_bytes:
                self.Log(f"Warning: Scaler key '{scaler_key}' not found in ObjectStore for date {current_date}. Model cannot be used.")
                # Keep previous model active if available, otherwise clear
                # self.active_scaler = None # Option: Clear if not found
                return
            loaded_scaler = pickle.loads(scaler_bytes)

            # Load Weights
            weights_bytes = self.ObjectStore.ReadBytes(weights_key)
            if not weights_bytes:
                self.Log(f"Warning: Weights key '{weights_key}' not found in ObjectStore for date {current_date}. Model cannot be used.")
                # Keep previous model active if available, otherwise clear
                # self.active_scaler = None # Option: Clear if not found
                return
            loaded_weights = pickle.loads(weights_bytes)

            # --- Successfully loaded ---
            self.active_scaler = loaded_scaler
            self.active_Wxh = loaded_weights.get('Wxh')
            self.active_Whh = loaded_weights.get('Whh')
            self.active_bh = loaded_weights.get('bh')
            self.active_Why = loaded_weights.get('Why')
            self.active_by = loaded_weights.get('by')
            self.model_loaded_for_date = target_training_end_date # Track the loaded model's date

            # *** FIX: Check if any weights are None explicitly ***
            # Old check causing error: if None in [self.active_Wxh, self.active_Whh, self.active_bh, self.active_Why, self.active_by]:
            if (self.active_Wxh is None or
                self.active_Whh is None or
                self.active_bh is None or
                self.active_Why is None or
                self.active_by is None):
                 self.Log(f"Error: Some weights were None after loading from key '{weights_key}'. Clearing active model.")
                 self.active_scaler = None # Clear model state
                 self.model_loaded_for_date = None
            else:
                 self.Log(f"Successfully loaded and activated model trained up to {target_training_end_date.strftime('%Y-%m-%d')}.")
                 self.Plot('Model Info', 'Model End Date Year', target_training_end_date.year)


        except Exception as e:
            self.Error(f"Critical error loading model/scaler for keys '{scaler_key}', '{weights_key}': {e}")
            self.active_scaler = None # Clear model state on error
            self.model_loaded_for_date = None


    def OnDailyReturnUpdated(self, sender, updated):
        """
        Event handler for the daily_returns ROC indicator.
        Updates the volatility indicator manually.
        Ref: QuantConnect RNN Strategy Analysis (Updated) - Section 1 (Indicator Chain)
        """
        if updated is not None and self.daily_returns.IsReady:
            self.volatility.Update(updated.EndTime, self.daily_returns.Current.Value)


    def forward(self, X):
        """
        Performs the forward pass of the SimpleRNN using NumPy.
        Uses the currently active model weights.
        Ref: QuantConnect RNN Strategy Analysis (Updated) - Section 1
        Args:
            X (np.array): Input sequence of shape (lookback, feature_count).

        Returns:
            float: The predicted output value, or None if weights are missing.
        """
        # *** FIX: Check if any weights are None explicitly ***
        if (self.active_Wxh is None or
            self.active_Whh is None or
            self.active_bh is None or
            self.active_Why is None or
            self.active_by is None):
             self.Error("Forward pass cannot execute: Active model weights are not fully loaded.")
             return None

        lookback, feature_count = X.shape
        hidden_size = self.active_bh.shape[0]

        # Initialize hidden state for this sequence
        h = np.zeros(hidden_size)

        # RNN layer computation
        for t in range(lookback):
            x_t = X[t]
            # Ensure arrays have compatible dimensions for dot product
            # Wxh: (feature_count, hidden_size)
            # Whh: (hidden_size, hidden_size)
            # bh: (hidden_size,)
            # Why: (hidden_size, 1)
            # by: (1,)
            term1 = np.dot(x_t, self.active_Wxh)       # (feature_count,) dot (feature_count, hidden_size) -> (hidden_size,)
            term2 = np.dot(h, self.active_Whh)         # (hidden_size,) dot (hidden_size, hidden_size) -> (hidden_size,)
            h = np.tanh(term1 + term2 + self.active_bh) # Add bias (hidden_size,)

        # Output layer computation
        # Why might need reshaping if it wasn't saved correctly, but assume (hidden_size, 1)
        # by should be scalar or (1,)
        y = np.dot(h, self.active_Why) + self.active_by # (hidden_size,) dot (hidden_size, 1) -> (1,) + (1,) -> (1,)

        # Ensure output is a scalar float
        return float(y[0]) if isinstance(y, (np.ndarray, list)) and len(y) > 0 else float(y)


    def OnData(self, data):
        '''OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.'''

        # --- Guards ---
        if self.IsWarmingUp:
            return

        # Check if SPY data is present
        if self.spy not in data or data[self.spy] is None:
            return

        # Check if a model (scaler and weights) is loaded and ready
        # Ref: Addressing Overfitting - Section 2 (Check model loaded)
        if self.active_scaler is None or self.model_loaded_for_date is None:
            self.Debug(f"{self.Time} - Waiting for applicable model to be loaded. Scaler ready: {self.active_scaler is not None}. Model date: {self.model_loaded_for_date}")
            # Attempt to load model if none is active - might happen on first day after warmup
            if self.model_loaded_for_date is None:
                self.UpdateModel()
                if self.active_scaler is None: # Check again after attempt
                    return
            else:
                return # Still no model

        # --- History & Feature Calculation ---
        try:
            # Request slightly more history to be safe with feature calculations involving shifts
            history_request_length = self.config["lookback"] + 5 # Added buffer
            history_df_raw = self.History(self.spy, history_request_length, Resolution.Daily)
            if history_df_raw.empty or len(history_df_raw) < self.config["lookback"] + 1:
                self.Debug(f"{self.Time} - Insufficient history ({len(history_df_raw)} days) for feature calculation (required {self.config['lookback'] + 1}).")
                return

             # Ensure correct columns and index using similar logic as research notebook
            if isinstance(history_df_raw.index, pd.MultiIndex):
                history_df = history_df_raw.droplevel(0)
            else:
                history_df = history_df_raw

            if isinstance(history_df.columns, pd.MultiIndex):
                 history_df.columns = history_df.columns.get_level_values(1)

            required_cols = ['open', 'close']
            if not all(col in history_df.columns for col in required_cols):
                 self.Debug(f"History DataFrame missing required columns: {required_cols}. Found: {history_df.columns.tolist()}")
                 return

            features_df = history_df[required_cols].copy()

            # Calculate features
            features_df['price_changes'] = features_df['close'] - features_df['open']
            features_df['overnight_gaps'] = features_df['open'] - features_df['close'].shift(1)

            # Select the exact lookback period AFTER calculating features (handles NaNs)
            # Take the most recent 'lookback' rows that are not NaN
            features_df.dropna(subset=['price_changes', 'overnight_gaps'], inplace=True)
            if len(features_df) < self.config["lookback"]:
                 self.Debug(f"{self.Time} - Insufficient non-NaN rows ({len(features_df)}) after feature calculation for lookback ({self.config['lookback']}).")
                 return

            features_raw = features_df[['price_changes', 'overnight_gaps']].iloc[-self.config["lookback"]:].values

            if features_raw.shape[0] != self.config["lookback"]:
                self.Debug(f"{self.Time} - Final feature array shape mismatch ({features_raw.shape}). Expected ({self.config['lookback']}, 2). Skipping prediction.")
                return
            if np.isnan(features_raw).any():
                 self.Debug(f"{self.Time} - NaN detected in final feature array. Skipping prediction.")
                 return


        except Exception as e:
            self.Error(f"Error during history fetching or feature calculation: {e}")
            import traceback
            self.Error(traceback.format_exc()) # Log full traceback for debugging
            return

        # --- Scaling ---
        try:
            features_scaled = self.active_scaler.transform(features_raw)
        except Exception as e:
            self.Error(f"Error applying scaler transform: {e}")
            return

        # --- Prediction ---
        try:
            self.prediction_value = self.forward(features_scaled)
            if self.prediction_value is None:
                 self.Error(f"{self.Time} - Prediction failed (forward pass returned None).")
                 return # Cannot proceed without prediction
            self.Plot('Prediction', 'Value', self.prediction_value)
        except Exception as e:
            self.Error(f"Error during forward pass execution: {e}")
            import traceback
            self.Error(traceback.format_exc()) # Log full traceback for debugging
            return

        # --- Plot Volatility ---
        if self.volatility.IsReady:
            current_volatility = self.volatility.Current.Value
            self.Plot('Volatility', 'Daily Volatility', current_volatility)
        else:
            current_volatility = None # Indicate not ready

        # --- Trading Logic ---
        # Ref: QuantConnect RNN Strategy Analysis (Updated) - Section 1 & 5 (Logic, Hysteresis, Filter)
        prediction_threshold = self.config["threshold"]
        volatility_threshold = self.config["volatility_threshold"]
        exit_days_required = self.config["exit_hysteresis_days"]

        is_invested = self.Portfolio[self.spy].IsLong

        # --- Update Exit Signal Counter ---
        if is_invested:
            if self.prediction_value <= prediction_threshold:
                self.exit_signal_days += 1
                self.Debug(f"{self.Time} - Exit signal received (Pred: {self.prediction_value:.4f} <= {prediction_threshold}). Consecutive days: {self.exit_signal_days}")
            else:
                # If signal reverses while invested, reset counter
                if self.exit_signal_days > 0:
                     self.Debug(f"{self.Time} - Exit signal reversed (Pred: {self.prediction_value:.4f} > {prediction_threshold}) while invested. Resetting counter.")
                     self.exit_signal_days = 0
        else:
            # Reset counter if not invested
            self.exit_signal_days = 0

        # --- Entry Logic ---
        if not is_invested and self.prediction_value > prediction_threshold:
            # Check volatility filter
            if current_volatility is not None and current_volatility < volatility_threshold:
                self.Debug(f"{self.Time} - Entry condition met (Pred: {self.prediction_value:.4f} > {prediction_threshold}). Volatility OK ({current_volatility:.4f} < {volatility_threshold}). Attempting SetHoldings.")
                self.SetHoldings(self.spy, 1.0)
                # Reset exit counter upon entry attempt
                self.exit_signal_days = 0
            elif current_volatility is None:
                 self.Debug(f"{self.Time} - Entry condition met (Pred: {self.prediction_value:.4f}) but Volatility indicator not ready. Skipping entry.")
            else:
                 self.Debug(f"{self.Time} - Entry condition met (Pred: {self.prediction_value:.4f}) but Volatility ({current_volatility:.4f}) >= threshold ({volatility_threshold}). Skipping entry.")

        # --- Exit Logic (Hysteresis) ---
        elif is_invested and self.exit_signal_days >= exit_days_required:
            self.Debug(f"{self.Time} - Exit condition met (Pred <= {prediction_threshold} for {self.exit_signal_days} days >= {exit_days_required}). Attempting Liquidate.")
            self.Liquidate(self.spy)
            # Reset exit counter after liquidation
            self.exit_signal_days = 0