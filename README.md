# GMF Investments Portfolio Optimization Using Time Series Forecasting

## Project Overview

This project is part of a challenge for 10 Academy's Artificial Intelligence Mastery program aimed at optimizing portfolio management strategies for Guide Me in Finance (GMF) Investments. The objective is to use time series forecasting on historical financial data to enhance portfolio performance, reduce risk, and leverage market opportunities. This solution integrates financial data analysis, forecasting models, and portfolio optimization techniques to help GMF make data-driven investment decisions.

## Business Objective

GMF Investments is a financial advisory firm specializing in personalized portfolio management through advanced time series forecasting models. By accurately predicting market trends, GMF aims to:
- Optimize asset allocation
- Minimize risk exposure
- Maximize portfolio returns for its clients

Financial analysts at GMF leverage real-time data and predictive insights to provide actionable recommendations for managing client portfolios, with a focus on high-risk and high-return stocks, bonds for stability, and diversified index funds.

## Project Structure

This repository contains the complete code, documentation, and analysis for the following stages:
- **Data Extraction**: Pulling historical financial data using the YFinance Python library.
- **Exploratory Data Analysis (EDA)**: Analyzing trends, seasonality, volatility, and other key indicators.
- **Time Series Forecasting Models**: Building ARIMA, SARIMA, and LSTM models to predict future trends.
- **Portfolio Optimization**: Using forecasted data to rebalance a sample portfolio of assets.

## Data

Historical data was collected for three primary assets:
1. **Tesla, Inc. (TSLA)** - High-growth stock with potential for high returns but significant volatility.
2. **Vanguard Total Bond Market ETF (BND)** - A bond ETF providing stability and income.
3. **S&P 500 ETF (SPY)** - An index fund representing the U.S. market for diversification.

Each dataset contains:
- **Price Metrics**: Open, High, Low, Close, Adjusted Close
- **Volume**: Total daily trades
- **Date Range**: January 1, 2015 - October 31, 2024

The financial data covers a 10-year period, allowing for robust time series analysis, pattern detection, and trend prediction.

## Methodology

1. **Data Preprocessing and Exploration**
   - Data cleansing, handling missing values, and outlier detection.
   - Exploratory analysis, including trend and seasonality decomposition, volatility analysis, and rolling averages.
  
2. **Time Series Forecasting Models**
   - **ARIMA**: Univariate model suitable for non-seasonal data.
   - **SARIMA**: Seasonally adjusted model for periodic data trends.
   - **LSTM**: Recurrent neural network model, leveraging long-term dependencies in time series.

   Each model was tuned using parameter optimization (grid search or auto_arima) and evaluated on:
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Percentage Error (MAPE)

3. **Forecast Analysis**
   - Historical and forecasted data are visualized with confidence intervals to highlight prediction accuracy.
   - Long-term trend analysis and volatility forecasts guide portfolio adjustment recommendations.

4. **Portfolio Optimization**
   - Defined an investment portfolio with three assets: TSLA (high-risk growth), BND (stable income), and SPY (diversified market exposure).
   - Forecasted returns and calculated the portfolio's expected return, volatility, Value at Risk (VaR), and Sharpe Ratio.
   - Rebalanced portfolio weights to maximize risk-adjusted return, informed by market forecasts and expected volatility.

## Results and Insights

- **Market Trends**: Key insights into the direction and volatility of TSLA, BND, and SPY, with actionable forecasts for the next 6-12 months.
- **Portfolio Optimization**: Adjustments in allocation to balance growth with risk management, supported by Sharpe Ratio and VaR metrics.
- **Model Performance**: Comparative analysis of model accuracy, highlighting SARIMA for seasonality-sensitive assets and LSTM for long-term stock trends.

## Deliverables

1. **Code**: All Python code is organized in Jupyter notebooks, with clear steps from data extraction to model training and evaluation.
2. **Documentation**: Detailed markdown cells within notebooks explain each step, methodology, and parameter choice.
3. **Forecast Visualizations**: Graphs displaying historical data, forecasts, and confidence intervals for each asset.
4. **Portfolio Analysis Report**: Summary of expected return, risk, and recommended asset allocation, with visualizations of risk-return profiles.

## Requirements

- **Python Libraries**: `yfinance`, `numpy`, `pandas`, `statsmodels`, `pmdarima`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow` (for LSTM models)
- **Development Tools**: Jupyter Notebook or any Python IDE for running the notebooks

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/amannnnnyyyy/Finance-Forecast-Portfolio-Optimizer.git
