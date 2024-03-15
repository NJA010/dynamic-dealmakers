import numpy as np


def get_simple_prices(products: dict, low: int, high: int):
    for product_name in products.keys():
        for uuid in products[product_name]:
            products[product_name][uuid] = np.round(np.random.uniform(low, high), 2)
    return products