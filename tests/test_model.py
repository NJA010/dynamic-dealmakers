import pytest
import pandas as pd
import numpy as np
from dynamic_pricing.model.simulator import price_calc, simulate_trades
from dynamic_pricing.utils import product_index
from scipy.optimize import minimize

np.random.seed(42)

@pytest.fixture()
def data():
    # Define number of products and time periods
    num_products = 10
    num_periods = 60
    num_teams = 4
    
    # Generate random data for selling price and quantity for each team
    data = []
    periods = pd.date_range(start='2024-01-01', periods=num_periods, freq='min')

    for team in range(num_teams):
        np.random.seed(team)  # Setting seed to ensure same data for each team
        selling_prices = np.random.uniform(0, 2, size=(num_products, num_periods))
        team_name = f"Team_{team+1}"
        for j, product_name in enumerate(product_index.keys()):
            for i, period in enumerate(periods):
                data.append(
                    [
                        team_name,
                        product_name,
                        period,
                        selling_prices[j, i],
                    ]
                )

    # Create DataFrame
    df = pd.DataFrame(
        data, columns=["team", "product_type", "time", "sell_price"]
    )

    return df


def test_data(data):
    params = {
        "a": np.zeros((10, 1)) + 20,
        "b": np.zeros((10, 1)) + 18,
        "c": np.zeros((10, 1)) - 30  # High for low stock
              }
    x0 = np.concatenate([params["a"], params["b"], params["c"]], axis=1).reshape(-1, )
    res = minimize(simulate_trades, x0, data, method="Nelder-Mead", tol=1e-6)
    # simulate_trades(data, x0, **params)
    assert True
