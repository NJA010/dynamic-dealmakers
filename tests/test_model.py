import pytest
import pandas as pd
import numpy as np


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
        'Date': np.repeat(dates, num_products),
        'Product': np.tile(products, num_periods),
        'Selling_Price': selling_prices.flatten(),
        'Quantity': quantities.flatten()
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    return df


def test_data(data):
    assert True
