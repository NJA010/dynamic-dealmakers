import pytest
from dotenv import load_dotenv

from dynamic_pricing.database import DatabaseClient, load_config
from dynamic_pricing.model.price_function import get_simple_prices, get_optimized_prices
import json
from copy import deepcopy
from dynamic_pricing.scrape import scrape, get_requests_headers, api_key, audience
import requests
from dynamic_pricing.utils import get_stock, products
load_dotenv()


@pytest.fixture()
def products():
    # Opening JSON file
    headers = get_requests_headers(api_key, audience)
    products_data = requests.get(f"{audience}/products", headers=headers).json()
    return products_data


@pytest.fixture()
def products_table():
    # Opening JSON file
    client = DatabaseClient(load_config())
    stock = get_stock(client)
    return stock


def test_get_simple_prices(products):
    prices = get_simple_prices(products, 1, 1)
    result = {}
    for product_name in products.keys():
        result[product_name] = {}
        for uuid in products[product_name]['products']:
            result[product_name][uuid] = 1.
    assert prices == result


def test_get_optimized_prices(products):
    # update with stock
    params = json.load(open('./tests/opt_params.json'))
    stock = {p: 50 for p in products}
    prices = get_optimized_prices(products, params=params, stock=stock)
    for product_name in prices.keys():
        for uuid in prices[product_name]:
            assert type(prices[product_name][uuid]) == float
