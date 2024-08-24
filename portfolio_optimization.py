"""
Portfolio Optimization Script
==============================

This script performs portfolio optimization using historical stock data 
downloaded from Yahoo Finance. It calculates the optimal portfolio 
allocation based on the Sharpe ratio and minimum volatility.

Features:
---------
1. Collects stock tickers from user input.
2. Downloads historical adjusted close prices for the specified stocks.
3. Calculates daily returns, mean returns, and the covariance matrix.
4. Generates a large number of random portfolios to simulate different 
   weight combinations.
5. Evaluates portfolio performance in terms of return, risk (volatility), 
   and the Sharpe ratio.
6. Identifies the portfolio with the maximum Sharpe ratio (best risk-adjusted return) 
   and the portfolio with the minimum volatility (lowest risk).
7. Visualizes the results in a scatter plot, highlighting the optimal portfolios.

How to Use:
-----------
1. Run the script.
2. Enter the stock tickers you want to include in your portfolio. Type 'end' 
   when you are done.
3. The script will download the historical data, perform the optimization, 
   and display a scatter plot of the portfolio performances.
4. The optimal portfolios will be printed in the console and highlighted in 
   the plot.

Requirements:
-------------
- numpy
- pandas
- yfinance
- matplotlib
- datetime

Example:
--------
Run the script and input stock tickers like:
AAPL, MSFT, GOOGL, AMZN
Type 'end' to finish input.

The script will then output the portfolio with the highest Sharpe ratio 
and the portfolio with the lowest volatility, and display a plot of all 
simulated portfolios.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.optimize as sco

# Function to get stock tickers from user
def get_stock_tickers():
    """
    Collects stock tickers from user input until 'end' is typed.
    Returns a sorted list of unique stock tickers.
    """
    stocks = []
    while True:
        ticker = input("What code do you want to add? (Type 'end' to finish): ").strip().upper()
        if ticker == "END":
            break
        if ticker:
            stocks.append(ticker)
    return sorted(set(stocks))

# Function to download historical stock data
def download_data(stocks, start_date, end_date):
    """
    Downloads historical adjusted close prices for the given stock tickers 
    between the specified start and end dates.
    
    Parameters:
    stocks (list): List of stock tickers.
    start_date (str): Start date in 'YYYY-MM-DD' format.
    end_date (str): End date in 'YYYY-MM-DD' format.
    
    Returns:
    DataFrame: Adjusted close prices for the given stock tickers.
    """
    try:
        data = yf.download(stocks, start=start_date, end=end_date)['Adj Close']
        data.sort_index(inplace=True)
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return pd.DataFrame()

# Function to calculate portfolio performance
def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    """
    Calculates the expected annual return and volatility of a portfolio.
    
    Parameters:
    weights (ndarray): Portfolio weights.
    mean_returns (Series): Mean daily returns of the stocks.
    cov_matrix (DataFrame): Covariance matrix of the stock returns.
    
    Returns:
    tuple: Expected annual return and volatility of the portfolio.
    """
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return portfolio_return, portfolio_std_dev

# Function to generate random portfolios
def generate_random_portfolios(num_portfolios, mean_returns, cov_matrix, num_stocks):
    """
    Generates random portfolios and calculates their performance.
    
    Parameters:
    num_portfolios (int): Number of portfolios to simulate.
    mean_returns (Series): Mean daily returns of the stocks.
    cov_matrix (DataFrame): Covariance matrix of the stock returns.
    num_stocks (int): Number of stocks in the portfolio.
    
    Returns:
    ndarray: Array containing the performance and weights of the simulated portfolios.
    """
    results = np.zeros((3 + num_stocks, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(num_stocks)
        weights /= np.sum(weights)
        portfolio_return, portfolio_std_dev = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std_dev
        results[2, i] = portfolio_return / portfolio_std_dev
        results[3:, i] = weights
    return results

# Function to plot results
def plot_results(results_frame, max_sharpe_port, min_vol_port, efficient_frontier_vol, efficient_frontier_ret):
    """
    Plots the results of the portfolio simulations.
    
    Parameters:
    results_frame (DataFrame): DataFrame containing the portfolio performance and weights.
    max_sharpe_port (Series): Portfolio with the maximum Sharpe ratio.
    min_vol_port (Series): Portfolio with the minimum volatility.
    efficient_frontier_vol (list): Volatility values of the efficient frontier.
    efficient_frontier_ret (ndarray): Return values of the efficient frontier.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(results_frame.stdev, results_frame.ret, c=results_frame.sharpe, marker='.', alpha=0.8, cmap='coolwarm')
    plt.plot(max_sharpe_port['stdev'], max_sharpe_port['ret'], 'y*', markersize=15.0)
    plt.plot(min_vol_port['stdev'], min_vol_port['ret'], 'r*', markersize=15.0)
    plt.plot(efficient_frontier_vol, efficient_frontier_ret, 'b', lw=2)  # Efficient frontier line
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()

# Main function
def main():
    """
    Main function to run the portfolio optimization and plot the results.
    """
    # Get stock tickers from user
    stocks = get_stock_tickers()
    if not stocks:
        print("No stocks selected.")
        return

    # Define the time period for historical data
    start_date = '2021-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')  # Use current date

    # Download historical data
    data = download_data(stocks, start_date=start_date, end_date=end_date)
    if data.empty:
        return

    # Calculate daily returns and their statistics
    returns = data.pct_change().dropna()
    mean_daily_returns = returns.mean()
    cov_matrix = returns.cov()

    # Generate random portfolios
    num_portfolios = 25000
    results = generate_random_portfolios(num_portfolios, mean_daily_returns, cov_matrix, len(stocks))

    # Create a DataFrame for the results
    results_frame = pd.DataFrame(results.T, columns=['ret', 'stdev', 'sharpe'] + stocks)

    # Identify the portfolios with the maximum Sharpe ratio and minimum volatility
    max_sharpe_port = results_frame.loc[results_frame['sharpe'].idxmax()]
    min_vol_port = results_frame.loc[results_frame['stdev'].idxmin()]

    # Define function to minimize portfolio volatility
    def min_func_volatility(weights):
        return calculate_portfolio_performance(weights, mean_daily_returns, cov_matrix)[1]

    # Calculate the efficient frontier
    target_returns = np.linspace(min_vol_port['ret'], max_sharpe_port['ret'], 100)
    efficient_frontier_vol = []
    for target in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: calculate_portfolio_performance(x, mean_daily_returns, cov_matrix)[0] - target}
        )
        bounds = tuple((0, 1) for _ in range(len(stocks)))
        result = sco.minimize(min_func_volatility, len(stocks) * [1. / len(stocks)], method='SLSQP', bounds=bounds, constraints=constraints)
        efficient_frontier_vol.append(result['fun'])
    
    # Plot the results
    plot_results(results_frame, max_sharpe_port, min_vol_port, efficient_frontier_vol, target_returns)

    # Print portfolio details
    print("Portfolio with maximum Sharpe Ratio:")
    print(max_sharpe_port)
    print("\nPortfolio with minimum volatility:")
    print(min_vol_port)

if __name__ == "__main__":
    main()
