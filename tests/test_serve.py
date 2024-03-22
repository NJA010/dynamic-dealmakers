import pytest

from dynamic_pricing.database import DatabaseClient, load_config
from dynamic_pricing.model.price_function import get_simple_prices
import json
from copy import deepcopy

from dynamic_pricing.utils import get_stock


@pytest.fixture()
def products():
    # Opening JSON file
    f = open('./tests/product_data.json')
    data = json.load(f)
    return data[0][2]


@pytest.fixture()
def products_table():
    # Opening JSON file
    client = DatabaseClient(load_config())
    stock = get_stock(client)
    return stock


def test_get_simple_prices(products):
    prices = get_simple_prices(products, 1, 1)
    result = deepcopy(products)
    for product_name in result.keys():
        for uuid in result[product_name]:
            result[product_name][uuid] = 1.
    assert prices == result