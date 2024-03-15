import pytest
from dynamic_pricing.model import get_simple_prices
import json
from copy import deepcopy


@pytest.fixture()
def products():
    # Opening JSON file
    f = open('./tests/products_format_example.json')
    data = json.load(f)
    return data


def test_get_simple_prices(products):
    prices = get_simple_prices(products, 1, 1)
    result = deepcopy(products)
    for product_name in result.keys():
        for uuid in result[product_name]:
            result[product_name][uuid] = 1.
    assert prices == result