import matplotlib.pyplot as plt
from scipy.stats import zscore
from statsmodels.tsa.seasonal import seasonal_decompose


def volatility(all_data):
    rolling_window = 20  # 20-day rolling window

    for asset, data in all_data.items():
        # Calculate rolling mean and standard deviation
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