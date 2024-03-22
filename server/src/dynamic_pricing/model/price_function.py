import numpy as np


def get_simple_prices(products: dict, low: int, high: int):
    result = {}
    for product_name in products.keys():
        result[product_name] = {}
        for uuid in products[product_name]['products']:
            result[product_name][uuid] = np.round(np.random.uniform(low, high), 2)
        print(result[product_name])
    return result


def price_function_sigmoid(
        stock: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray
):
    """
    For a given stock (x), obtain price (y) given parameters a, b, c for each product type
    :param stock:
    :param a: horizontal transformation (mean), a > 0
    :param b: Increasing starting price value. b > 0
    :param c: shape transformation
    :return: price vector for each product type
    """
    return b / (1 + np.exp((-stock+a)/c))


if __name__ == "__main__":
    from dynamic_pricing.scrape import get_requests_headers, api_key, audience
    import requests

    headers = get_requests_headers(api_key, audience)
    product_data = requests.get(f"{audience}/products", headers=headers).json()
    
    get_simple_prices(products=product_data, low=3, high=100)