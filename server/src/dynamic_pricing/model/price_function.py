import numpy as np


def get_simple_prices(products: dict, low: int, high: int):
    for product_name in products.keys():
        for uuid in products[product_name]:
            print(products[product_name][uuid])
            products[product_name][uuid] = np.round(np.random.uniform(low, high), 2)
    return products


def price_function_sigmoid(
        stock: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray
):
    """
    For a given stock (x), obtain price (y) given parameters a, b, c for each product type
    :param stock:
    :param a: horizontal transformation
    :param b: vertical transformation
    :param c: shape transformation
    :return: price vector for each product type
    """
    return c / (1 + np.exp(-stock+a)) + b


if __name__ == "__main__":
    from dynamic_pricing.scrape import get_requests_headers, api_key, audience
    import requests

    headers = get_requests_headers(api_key, audience)
    product_data = requests.get(f"{audience}/products", headers=headers).json()
    
    get_simple_prices(products=product_data, low=3, high=100)