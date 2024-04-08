import pytest

from dynamic_pricing.database import DatabaseClient, load_config
from dynamic_pricing.model.price_function import get_simple_prices, get_optimized_prices
import json

@pytest.fixture()
def products():
    f = open('./tests/product_data.json')
    data = json.load(f)
    return data[0][2]


def test_get_simple_prices(products):
    prices = get_simple_prices(products, 1, 1)
    result = {}
    for product_name in products.keys():
        result[product_name] = {}
        for uuid in products[product_name]['products']:
            result[product_name][uuid] = 1.
    assert prices == result


# def test_get_optimized_prices(products):
#     # update with stock
#     params = json.load(open('./tests/opt_params.json'))
#     stocks_data = json.load(open('./tests/stock_format_example.json'))
#
#     db = DatabaseClient(load_config())
#
#     params = db.read("SELECT * from simulation where total_revenue = (select max(total_revenue) from simulation)")
#     params_formatted = {row[2]: [float(p) for p in row[3]] for row in params}
#     prices = get_optimized_prices(products, stocks_data, params_formatted)
#     simp_prices = get_simple_prices(products, 1, 1)
#
#     for product_name in prices.keys():
#         for uuid in prices[product_name]:
#             assert type(prices[product_name][uuid]) == type(simp_prices[product_name][uuid])
