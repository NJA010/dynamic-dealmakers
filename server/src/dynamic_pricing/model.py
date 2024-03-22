import numpy as np


def get_simple_prices(products: dict, low: int, high: int):
    for product_name in products.keys():
        for uuid in products[product_name]:
            products[product_name][uuid] = np.round(np.random.uniform(low, high), 2)
    return products

if __name__ == "__main__":
    from dynamic_pricing.scrape import scrape, get_requests_headers, api_key, audience
    import requests

    headers = get_requests_headers(api_key, audience)
    product_data = requests.get(f"{audience}/products", headers=headers).json()
    
    get_simple_prices(products=product_data, low=3, high=100)