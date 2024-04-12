import numpy as np
import jax.numpy as jnp


def get_simple_prices(products: dict, low: int, high: int):
    result = {}
    for product_name in products.keys():
        result[product_name] = {}
        for uuid in products[product_name]['products']:
            result[product_name][uuid] = np.round(np.random.uniform(low, high), 2)
    return result


def get_optimized_prices(products: dict, stock: dict, params: dict):
    """
    calculate prices per product type based on stock and sigmoid parameters
    :param products:
    :param stock: {product: stock}
    :param params: {product: [params]}
    :return:
    """
    result = {}
    product_stock = {}
    product_batch = {}
    # obtain batch_id -> product_name map
    for product_name in products.keys():
        result[product_name] = {}
        for uuid, value in products[product_name]['products'].items():
            product_batch[str(value['id'])] = product_name
    # obtain product_name -> total stock map
    for batch_id, s in stock.items():
        if product_batch[batch_id] in product_stock.keys():
            product_stock[str(product_batch[batch_id])] += s
        else:
            product_stock[str(product_batch[batch_id])] = s

    for product_name in products.keys():
        result[product_name] = {}
        for uuid, value in products[product_name]['products'].items():
            result[product_name][uuid] = float(price_function_sigmoid(
                product_stock[product_name], *params[product_name]
            ))
    return result


def price_function_sigmoid(
    stock: jnp.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
):
    """
    For a given stock (x), obtain price (y) given parameters a, b, c for each product type
    :param stock:
    :param a: horizontal transformation (mean), a > 0
    :param b: Increasing starting price value. b > 0
    :param c: shape transformation
    :return: price vector for each product type
    """
    # data matrix is mixed so ensure stock is float here
    # stock = stock.astype(float)
    return b / (1 + jnp.exp(-jnp.divide(stock - a, c))) + 1


if __name__ == "__main__":
    from dynamic_pricing.scrape import get_requests_headers, api_key, audience
    from dynamic_pricing.env_setting import define_app_creds
    import requests

    _ = define_app_creds()

    headers = get_requests_headers(api_key, audience)
    product_data = requests.get(f"{audience}/products", headers=headers).json()

    get_simple_prices(products=product_data, low=3, high=100)
