import pytest
import pandas as pd
import numpy as np
from dynamic_pricing.model.revenue import revenue_calc
from dynamic_pricing.model.price_function import price_function_sigmoid
from dynamic_pricing.utils import get_hardcoded_sigmoid_params

products = ['apples-red',
            'apples-green',
            'bananas',
            'bananas-organic',
            'broccoli',
            'rice',
            'wine',
            'cheese',
            'beef',
            'avocado'
            ]


@pytest.fixture()
def data():
    # Define number of products and time periods
    num_products = 10
    num_periods = 60

    # Generate random data for selling price and quantity
    np.random.seed(0)
    selling_prices = np.random.randint(2, 20, size=(num_products, num_periods))
    quantities = np.random.randint(1, 20, size=(num_products, num_periods))

    # Generate dates for each time period
    dates = pd.date_range(start='2024-01-01', periods=num_periods, freq='min')

    # Create a list of product names
    products = [f"product_{i + 1}" for i in range(num_products)]

    # Create a dictionary to store the data
    data = {
        'time': np.repeat(dates, num_products),
        'product': np.tile(products, num_periods),
        'sell_price': selling_prices.flatten(),
        'quantity': quantities.flatten()
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    return df


def test_data(data):
    params = {"a": np.zeros((len(data), 1)) + 25,
              "b": np.zeros((len(data), 1)) + 10,
              "c": np.zeros((len(data), 1)) - 28  # High for low stock
              }
    revenue_calc(data, price_function_sigmoid, **params)
    assert True
