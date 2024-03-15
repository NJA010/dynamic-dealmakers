# from dynamic_pricing.price import Price
from dynamic_pricing.scrape import scrape, get_requests_headers, api_key, audience
from dynamic_pricing.model import get_simple_prices
from flask import Flask
from dotenv import load_dotenv
import requests

load_dotenv()
app = Flask(__name__)


# @app.route('/pricing', methods=['GET'])
# def pricing():
#     price = Price()
#     return "300 euro voor een banaan"

@app.route('/scrape-data', methods=['GET'])
def pricing():
    scrape()
    return "Data scraped!"


@app.route('/prices', methods=['GET'])
def get_prices():
    headers = get_requests_headers(api_key, audience)
    product_data = requests.get(f"{audience}/products", headers=headers).json()

    result = get_simple_prices(product_data, low=3, high=100)
    return result


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)