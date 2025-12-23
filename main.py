import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from portfolio_lib import (
    get_data, calculate_metrics, simulate_efficient_frontier,
    get_max_sharpe_ratio, get_min_volatility, get_hrp_allocation,
    portfolio_performance
)

# 1. Define Tickers
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'JNJ', 'V', 'PG', 'NVDA']
print(f"Downloading data for: {tickers}")

# 2. Get Data
try:
    data = get_data(tickers, start_date="2020-01-01", end_date="2023-01-01")
    if data.empty:
        print("Error: No data downloaded. Check tickers or internet connection.")
        exit()
except Exception as e:
    print(f"Error downloading data: {e}")
    exit()

# 3. Calculate Metrics
returns, mean_returns, cov_matrix = calculate_metrics(data)

# 4. Markowitz Optimization (Monte Carlo)
print("Running Monte Carlo Simulation...")
num_portfolios = 20000
results, weights_record = simulate_efficient_frontier(mean_returns, cov_matrix, num_portfolios)

# Find Optimal Portfolios
max_sharpe_vol, max_sharpe_ret, max_sharpe_alloc = get_max_sharpe_ratio(results, weights_record)
min_vol_vol, min_vol_ret, min_vol_alloc = get_min_volatility(results, weights_record)

print("-" * 30)
print("Max Sharpe Ratio Portfolio:")
print(f"Return: {max_sharpe_ret:.2f}, Volatility: {max_sharpe_vol:.2f}")
print(pd.Series(max_sharpe_alloc, index=tickers).round(4))

print("-" * 30)
print("Min Volatility Portfolio:")
print(f"Return: {min_vol_ret:.2f}, Volatility: {min_vol_vol:.2f}")
print(pd.Series(min_vol_alloc, index=tickers).round(4))

# 5. Hierarchical Risk Parity (HRP)
print("-" * 30)
print("Running HRP...")
hrp_alloc = get_hrp_allocation(returns)
hrp_ret, hrp_vol = portfolio_performance(hrp_alloc.values, mean_returns, cov_matrix)

print("HRP Portfolio:")
print(f"Return: {hrp_ret:.2f}, Volatility: {hrp_vol:.2f}")
print(hrp_alloc.round(4))

# 6. Visualization
plt.figure(figsize=(12, 8))
plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='viridis', marker='o', s=10, alpha=0.3)
plt.colorbar(label='Sharpe Ratio')
plt.scatter(max_sharpe_vol, max_sharpe_ret, marker='*', color='r', s=500, label='Max Sharpe')
plt.scatter(min_vol_vol, min_vol_ret, marker='*', color='b', s=500, label='Min Volatility')
plt.scatter(hrp_vol, hrp_ret, marker='*', color='g', s=500, label='HRP')

plt.title('Efficient Frontier vs HRP')
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Return')
plt.legend(labelspacing=0.8)
plt.grid(True, linestyle='--', alpha=0.6)

# Save plot
plt.savefig('efficient_frontier_hrp.png')
print("\nPlot saved as 'efficient_frontier_hrp.png'")
# plt.show() # Uncomment if running locally with display
