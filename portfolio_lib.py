import yfinance as yf
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.optimize import minimize

def get_data(tickers, start_date="2020-01-01", end_date="2023-01-01"):
    """
    Download adjusted close prices for the given tickers.
    """
    data = yf.download(tickers, start=start_date, end=end_date)
    if 'Adj Close' in data.columns:
        return data['Adj Close']
    elif 'Close' in data.columns:
        return data['Close']
    else:
        raise ValueError("No price data found (neither 'Adj Close' nor 'Close')")

def calculate_metrics(data):
    """
    Calculate mean returns and covariance matrix.
    """
    returns = data.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    return returns, mean_returns, cov_matrix

def portfolio_performance(weights, mean_returns, cov_matrix):
    """
    Calculate portfolio return and volatility.
    """
    returns = np.sum(mean_returns * weights)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, std

def simulate_efficient_frontier(mean_returns, cov_matrix, num_portfolios=10000, risk_free_rate=0.02):
    """
    Simulate random portfolios to generate the Efficient Frontier.
    """
    results = np.zeros((3, num_portfolios))
    weights_record = []
    num_assets = len(mean_returns)

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        portfolio_return, portfolio_std = portfolio_performance(weights, mean_returns, cov_matrix)
        
        results[0,i] = portfolio_std
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std # Sharpe Ratio

    return results, weights_record

def get_max_sharpe_ratio(results, weights_record):
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
    max_sharpe_allocation = weights_record[max_sharpe_idx]
    return sdp, rp, max_sharpe_allocation

def get_min_volatility(results, weights_record):
    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0,min_vol_idx], results[1,min_vol_idx]
    min_vol_allocation = weights_record[min_vol_idx]
    return sdp_min, rp_min, min_vol_allocation

# --- HRP Implementation ---

def getIVP(cov, **kargs):
    """
    Compute the inverse-variance portfolio.
    """
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp

def getClusterVar(cov, cItems):
    """
    Compute variance per cluster.
    """
    cov_slice = cov.loc[cItems, cItems] # pandas indices
    w = getIVP(cov_slice).reshape(-1, 1)
    cVar = np.dot(np.dot(w.T, cov_slice), w)[0, 0]
    return cVar

def getQuasiDiag(link):
    """
    Sort clustered items by distance.
    """
    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    numItems = link[-1, 3] # number of original items
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2) # make space
        df0 = sortIx[sortIx >= numItems] # find clusters
        i = df0.index
        j = df0.values - numItems
        sortIx[i] = link[j, 0] # item 1
        df0 = pd.Series(link[j, 1], index=i + 1)
        sortIx = pd.concat([sortIx, df0]) # item 2
        sortIx = sortIx.sort_index()
        sortIx.index = range(sortIx.shape[0]) # re-index
    return sortIx.tolist()

def getRecBisection(cov, sortIx):
    """
    Compute HRP alloc.
    """
    w = pd.Series(1, index=sortIx)
    cItems = [sortIx] # initialize all items in one cluster
    while len(cItems) > 0:
        cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1] # bi-section
        for i in range(0, len(cItems), 2): # parse in pairs
            cItems0 = cItems[i] # cluster 1
            cItems1 = cItems[i + 1] # cluster 2
            cVar0 = getClusterVar(cov, cItems0)
            cVar1 = getClusterVar(cov, cItems1)
            alpha = 1 - cVar0 / (cVar0 + cVar1)
            w[cItems0] *= alpha # weight 1
            w[cItems1] *= 1 - alpha # weight 2
    return w

def get_hrp_allocation(returns):
    """
    Main HRP function.
    """
    cov = returns.cov()
    corr = returns.corr()
    dist = (0.5 * (1 - corr))**0.5
    link = sch.linkage(dist, 'single')
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist() # recover labels
    hrp = getRecBisection(cov, sortIx)
    return hrp.sort_index()
