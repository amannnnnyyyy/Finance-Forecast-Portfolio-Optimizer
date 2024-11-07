def calc_daily_return(all_data):
    for asset, data in all_data.items():
        data['Daily Return'] = data['Adj Close'].pct_change() * 100 
        data.dropna(inplace=True)
    
def VAR(asset,data):
    VaR_95 = data['Daily Return'].quantile(0.05)  # 5th percentile for 95% confidence
    print(f"{asset} - 5% Value at Risk (95% confidence): {VaR_95:.4f}")
    return VaR_95

def Sharpe(asset,data):
    risk_free_rate = 0.02  # 2% annual risk-free rate
    excess_return = data['Daily Return'].mean() - risk_free_rate / 252  # Daily excess return
    sharpe_ratio = excess_return / data['Daily Return'].std()  # Daily Sharpe Ratio
    print(f"{asset} Sharpe Ratio: {sharpe_ratio:.4f}")
    return sharpe_ratio