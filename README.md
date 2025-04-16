# QuantConnect Project Setup

This project contains two primary files for algorithm development and research analysis on the QuantConnect Cloud Platform:

- `main.py`: Python algorithm script for backtesting and live trading.
- `research.ipynb`: Jupyter Notebook for exploratory data analysis and research.

## Strategy Overview

The trading model implemented in `main.py` is a **universe selection and alpha generation strategy** based on fundamental factors and earnings data. Key features include:

- **Universe Selection**: The algorithm dynamically selects U.S. equities based on fundamental filters, such as positive ROE, EPS growth, and price-to-sales ratios.
- **Alpha Generation**: It uses a ranking mechanism that considers multiple custom fundamental factors to identify the top 10 long and bottom 10 short stocks.
- **Portfolio Construction**: An equal-weighting scheme is applied across selected securities.
- **Execution Model**: Immediate execution of insights using QuantConnect’s built-in models.

This is a long-short equity model with a research-first design — first explored and analyzed in `research.ipynb`, then implemented as a systematic strategy in `main.py`.

The `research.ipynb` file trains a **Simple Recurrent Neural Network (RNN)** using historical SPY data to predict next-day price changes based on price movements and overnight gaps. Key aspects include:

- **Feature Engineering**: Calculates price changes and overnight gaps from OHLCV data.
- **Preprocessing**: Scales the features using `StandardScaler` and constructs lookback windows for the RNN.
- **Model Architecture**: Uses `SimpleRNN`, `Dropout`, and `Dense` layers with L2 regularization.
- **Training Logic**: Includes early stopping and validation splitting.
- **Deployment**: Saves trained weights and scalers to QuantConnect’s `ObjectStore` for use in production models.

This notebook is used to pre-train the RNN, whose outputs (weights and scaler) can be directly used within `main.py` to inform alpha signals, making it a hybrid **quantamental + deep learning** pipeline.

## Setup Instructions

Follow these steps to set up and run this project on QuantConnect:

### Step 1: Create a Project

1. Navigate to [QuantConnect](https://www.quantconnect.com/) and sign into your account.
2. Create a new project:
   - Click "Projects" → "Create Project."
   - Provide a descriptive name and select Python as your language.

### Step 2: Upload Files

1. Open your newly created project.
2. Click on the "Explorer" tab in the left sidebar.
3. Select "Upload" and upload both `main.py` and `research.ipynb`.

### Step 3: Running the Algorithm (`main.py`)

- Ensure that `main.py` is correctly uploaded to your project.
- Set your algorithm parameters (dates, initial capital, etc.) directly within the script or via the QuantConnect interface.
- Click the "Build" button to ensure there are no syntax errors.
- Click "Backtest" to run your algorithm.

### Step 4: Conducting Research (`research.ipynb`)

- Navigate to the "Research" tab within your project.
- Open `research.ipynb`.
- Run each cell sequentially for your exploratory analysis and model training.
- Upon completion, the notebook will save the trained model and scaler to QuantConnect’s `ObjectStore`.

## Requirements and Dependencies

QuantConnect Cloud provides a fully managed environment with pre-installed Python packages like:

- numpy
- pandas
- matplotlib
- sklearn
- tensorflow (with Keras)

If your notebook or script requires additional libraries, ensure compatibility with QuantConnect's environment or use alternative supported methods.

## Support

For any issues, visit the [QuantConnect Documentation](https://www.quantconnect.com/docs/home/home) or utilize the community forums for assistance.

