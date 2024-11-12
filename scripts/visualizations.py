import matplotlib.pyplot as plt
from scipy.stats import zscore
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
import numpy as np


def volatility(all_data):
    rolling_window = 20  

    for asset, data in all_data.items():
        data['Rolling Mean'] = data['Adj Close'].rolling(window=rolling_window).mean()
        data['Rolling Std Dev'] = data['Adj Close'].rolling(window=rolling_window).std()

        # Plot Adjusted Close Price with rolling mean and standard deviation
        plt.figure(figsize=(10, 6))
        plt.plot(data['Adj Close'], label=f'{asset.upper()} Adjusted Close Price')
        plt.plot(data['Rolling Mean'], label=f'{asset.upper()} {rolling_window}-Day Rolling Mean', linestyle='--')
        plt.fill_between(data.index, 
                        data['Rolling Mean'] - 2 * data['Rolling Std Dev'], 
                        data['Rolling Mean'] + 2 * data['Rolling Std Dev'], 
                        color='gray', alpha=0.3, label=f'{asset.upper()} 2x Std Dev')
        plt.title(f'{asset.upper()} Volatility with Rolling Mean and Std Dev')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.show()


def detect_outliers(all_data):
    # Detect outliers using Z-score
    for asset, data in all_data.items():
        data['Z-Score'] = zscore(data['Adj Close'])
        
        # Define outliers as values with Z-score > 3 or < -3
        outliers = data[data['Z-Score'].abs() > 3]
        
        # Plot Adjusted Close Price with outliers
        plt.figure(figsize=(10, 6))
        plt.plot(data['Adj Close'], label=f'{asset.upper()} Adjusted Close Price')
        plt.scatter(outliers.index, outliers['Adj Close'], color='red', label='Outliers', zorder=5)
        plt.title(f'{asset.upper()} Outliers in Adjusted Close Price')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Print outliers' dates and values
        print(f"Outliers for {asset.upper()}:")
        print(outliers[['Adj Close', 'Z-Score']])
        print("\n")


def plot_daily_percentage(all_data):
    for asset, data in all_data.items():
        plt.figure(figsize=(10, 6))
        plt.plot(data['Daily Return'], label=f'{asset.upper()} Daily Return')
        plt.title(f'{asset.upper()} Daily Percentage Change')
        plt.xlabel('Date')
        plt.ylabel('Daily Return (%)')
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_significant_anomalies(all_data):
    threshold = 5  # 5% threshold for high/low returns

    for asset, data in all_data.items():
        high_returns = data[data['Daily Return'] > threshold]
        low_returns = data[data['Daily Return'] < -threshold]

        # Plot high and low returns
        plt.figure(figsize=(10, 6))
        plt.plot(data['Daily Return'], label=f'{asset.upper()} Daily Return')
        plt.scatter(high_returns.index, high_returns['Daily Return'], color='green', label='High Returns', zorder=5)
        plt.scatter(low_returns.index, low_returns['Daily Return'], color='red', label='Low Returns', zorder=5)
        plt.title(f'{asset.upper()} Days with Unusually High/Low Returns')
        plt.xlabel('Date')
        plt.ylabel('Daily Return (%)')
        plt.legend()
        plt.grid(True)
        plt.show()

        print(f"High Returns for {asset.upper()}:")
        print(high_returns[['Daily Return']])
        print("\n")
        
        print(f"Low Returns for {asset.upper()}:")
        print(low_returns[['Daily Return']])
        print("\n")

def decomposition(all_data):
    for asset, data in all_data.items():
        decomposition = seasonal_decompose(data['Adj Close'], model='additive', period=252)
        
        fig = decomposition.plot()
        
        fig.suptitle(f"Seasonal Decomposition of Adjusted Close for {asset.upper()}", fontsize=16, y=1.05)
        
        plt.tight_layout()
        
        plt.show()

def volatility_rolling(window_size,all_data):
    # Analyze volatility for each asset
    for asset, data in all_data.items():
        # Calculate the rolling mean and standard deviation for the adjusted close price
        rolling_mean = data['Adj Close'].rolling(window=window_size).mean()
        rolling_std = data['Adj Close'].rolling(window=window_size).std()

        # Plot the adjusted close price along with rolling mean and rolling standard deviation
        plt.figure(figsize=(12, 8))

        # Plot the adjusted close price
        plt.subplot(311)
        plt.plot(data['Adj Close'], label=f'{asset.upper()} Adjusted Close', color='blue')
        plt.title(f'{asset.upper()} - Adjusted Close Price')
        plt.legend()

        # Plot the rolling mean
        plt.subplot(312)
        plt.plot(rolling_mean, label=f'{asset.upper()} {window_size}-Day Rolling Mean', color='orange')
        plt.title(f'{asset.upper()} - {window_size}-Day Rolling Mean')
        plt.legend()

        # Plot the rolling standard deviation
        plt.subplot(313)
        plt.plot(rolling_std, label=f'{asset.upper()} {window_size}-Day Rolling Std Dev', color='red')
        plt.title(f'{asset.upper()} - {window_size}-Day Rolling Std Dev (Volatility)')
        plt.legend()

        plt.tight_layout()
        plt.show()



def VAR_plot(VaR_values):
    if VaR_values:  # Check if there are any valid VaR values to plot
        plt.figure(figsize=(10, 6))
        
        # Ensure keys and values are in the correct format
        keys = list(VaR_values.keys())
        values = list(VaR_values.values())
        
        plt.bar(keys, values, color='orange')
        plt.title('5% Value at Risk (VaR) for Different Assets')
        plt.xlabel('Assets')
        plt.ylabel('Value at Risk (VaR)')
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # line at zero for reference
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("No valid VaR values to plot.")

def sharpe_plot(sharpe_values):
    if sharpe_values:
        plt.figure(figsize=(10, 6))

        keys = list(sharpe_values.keys())
        values = list(sharpe_values.values())

        plt.bar(keys, values, color='blue')
        plt.title('Sharpe Ratio for Different Assets')
        plt.xlabel('Assets')
        plt.ylabel('Sharpe Ratio')
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--') 
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("No valid Sharpe ratios to plot.")


def correlation_returns(daily_returns):
    # Calculate the correlation matrix
    corr_matrix = daily_returns.corr()

    # Plotting the correlation matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix Heatmap')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.show()


def covariance_returns(daily_returns):
    # Calculate the covariance matrix
    cov_matrix = daily_returns.cov()


    plt.figure(figsize=(8, 6))
    sns.heatmap(cov_matrix, annot=True, fmt=".8f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title('Covariance Matrix Heatmap')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.show()


def daily_plot_VaR(df,var_Tesla):
    plt.figure(figsize=(10, 6))
    plt.hist(df['TSLA_daily_return'].dropna(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(var_Tesla, color='red', linestyle='dashed', linewidth=2, label=f'VaR (95%): {var_Tesla:.4f}')
    plt.title("Tesla's Daily Returns Distribution with VaR at 95% Confidence")
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


def cumulative_return_plot(cumulative_return):
    # Plot cumulative returns
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_return, label="Portfolio Cumulative Return", color='blue')
    plt.title("Portfolio Cumulative Returns Over Time")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return ($1 Investment)")
    plt.legend()
    plt.show()


def daily_annual_sharpe_ratio(sharpe_ratios):
    # Plot the Sharpe ratios
    plt.figure(figsize=(8, 5))
    plt.bar(sharpe_ratios.keys(), sharpe_ratios.values(), color=['coral', 'dodgerblue'])
    plt.title("Comparison of Daily and Annualized Sharpe Ratios")
    plt.ylabel("Sharpe Ratio")
    plt.ylim(0, max(sharpe_ratios.values()) * 1.2)
    plt.show()


def montecarlo_simulation(df,mean_returns,cov_matrix,optimized_weights):
    dataset_size = len(df)  # Number of rows (days) in your dataset
    num_days = len(df)  # The number of data points (days) in the dataset

    num_simulations = min(1000, int(dataset_size / 10))

    simulated_portfolios = np.zeros((num_simulations, num_days))

    for i in range(num_simulations):
        # Generate random returns for each asset (TSLA, BND, SPY)
        random_returns = np.random.multivariate_normal(mean_returns, cov_matrix, num_days)
        
        # Calculate the portfolio returns for each day (weighted sum of returns)
        simulated_portfolios[i] = np.dot(random_returns, optimized_weights)

    # Plot the simulated portfolio returns
    plt.figure(figsize=(10, 6))
    plt.plot(simulated_portfolios.T, color='blue', alpha=0.1)
    plt.title('Monte Carlo Simulation: Simulated Portfolio Performance')
    plt.xlabel('Days')
    plt.ylabel('Portfolio Daily Return')
    plt.show()

    # Calculate the cumulative return for each simulation to see the total portfolio growth over time
    cumulative_returns = np.cumsum(simulated_portfolios, axis=1)

    # Plot the cumulative returns to visualize portfolio growth over time
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns.T, color='blue', alpha=0.1)
    plt.title('Monte Carlo Simulation: Cumulative Portfolio Return')
    plt.xlabel('Days')
    plt.ylabel('Cumulative Portfolio Return')
    plt.show()

def cumulative_returns_indiv_assets(df, weighted_daily_return):
    cumulative_returns = (1 + weighted_daily_return).cumprod()
    cumulative_returns_TESLA = (1 + df['TSLA_daily_return']).cumprod()
    cumulative_returns_BND = (1 + df['BND_daily_return']).cumprod()
    cumulative_returns_SPY = (1 + df['SPY_daily_return']).cumprod()

    # Plot cumulative returns
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns, label="Optimized Portfolio", color='blue')
    plt.plot(cumulative_returns_TESLA, label="Tesla (TSLA)", color='red')
    plt.plot(cumulative_returns_BND, label="Bond (BND)", color='green')
    plt.plot(cumulative_returns_SPY, label="S&P 500 (SPY)", color='orange')

    plt.title("Cumulative Returns of Portfolio and Individual Assets")
    plt.xlabel("Days")
    plt.ylabel("Cumulative Return")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()

def risk_return_analysis(df, mean_returns, average_portfolio_return, portfolio_volatility):
    # Plot Risk vs Return for the assets and portfolio
    returns = [mean_returns['TSLA_daily_return'], mean_returns['BND_daily_return'], mean_returns['SPY_daily_return'], average_portfolio_return]
    volatility = [df['TSLA_daily_return'].std(), df['BND_daily_return'].std(), df['SPY_daily_return'].std(), portfolio_volatility]

    plt.figure(figsize=(10, 6))

    # Scatter plot for individual assets with separate colors and labels
    plt.scatter(volatility[0], returns[0], color='red', label='Tesla', s=100)
    plt.scatter(volatility[1], returns[1], color='green', label='Bond', s=100)
    plt.scatter(volatility[2], returns[2], color='orange', label='SPY', s=100)

    # Scatter plot for the optimized portfolio
    plt.scatter(portfolio_volatility, average_portfolio_return, color='blue', label="Optimized Portfolio", marker='x', s=100)

    # Add labels and legend
    for i, txt in enumerate(['TSLA', 'BND', 'SPY', 'Portfolio']):
        plt.annotate(txt, (volatility[i], returns[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.title("Risk vs Return Analysis")
    plt.xlabel("Volatility (Risk)")
    plt.ylabel("Expected Return")
    plt.legend()
    plt.grid(True)
    plt.show()
