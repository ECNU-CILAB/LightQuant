import numpy as np
import torch
from sklearn.metrics import accuracy_score
np.seterr(divide='ignore', invalid='ignore')


def calculate_MDD(asset_list):
    max_asset = np.max(asset_list)
    drawdowns = [(max_asset - value) / max_asset for value in asset_list]
    MDD = np.max(drawdowns)
    return MDD


def calculate_Calmar_Ratio(ARR, MDD):
    return ARR / MDD


def calculate_IR(asset_list, risk_free_rate=0.03):
    # Calculate returns and avoid dividing by zero
    returns = np.diff(asset_list) / np.where(asset_list[:-1] != 0, asset_list[:-1], 1)  # Prevent division by zero
    # Ignore NaN and Inf values by removing them
    returns = returns[~np.isnan(returns) & ~np.isinf(returns)]

    if len(returns) == 0:
        return np.nan  # If no valid returns, return NaN

    excess_returns = returns - risk_free_rate / 252  # Assuming daily returns
    annualized_excess_return = np.mean(excess_returns) * 252  # Annualize
    annualized_volatility = np.std(excess_returns) * np.sqrt(252)  # Annualize
    if annualized_volatility == 0:
        return np.nan  # If volatility is zero, return NaN
    IR = annualized_excess_return / annualized_volatility
    return IR


def calculate_ACC(actual_directions, predicted_directions):

    return accuracy_score(actual_directions, predicted_directions)

def calculate_ARR(asset_list):
    if not asset_list or len(asset_list) < 2:
        print(f"[Warning] Asset list length:{len(asset_list)}")
        return float('nan')
    initial = asset_list[0]
    final = asset_list[-1]
    n_days = len(asset_list)
    return (final / initial) ** (252 / n_days) - 1


def calculate_SR(asset_list, risk_free_rate=0.01):

    if len(asset_list) < 2:
        return np.nan


    returns = np.diff(asset_list) / asset_list[:-1]


    annualized_return = np.mean(returns) * 252


    annualized_volatility = np.std(returns) * np.sqrt(252)


    excess_return = annualized_return - risk_free_rate


    if annualized_volatility == 0:
        return np.nan

    sharpe_ratio = excess_return / annualized_volatility

    return sharpe_ratio


def calculate_cumulative_return(asset_list):

    if not asset_list or len(asset_list) < 2:
        print(f"[Warning] Asset list length:{len(asset_list)}")
        return []

    initial = asset_list[0]
    cumulative_returns = []

    for i in range(len(asset_list)):
        cumulative_return = (asset_list[i] / initial) - 1
        cumulative_returns.append(cumulative_return)

    return cumulative_returns