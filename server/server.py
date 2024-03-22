# from dynamic_pricing.price import Price
from dynamic_pricing.scrape import scrape, get_requests_headers, api_key, audience
from dynamic_pricing.model import get_simple_prices
from flask import Flask
from dotenv import load_dotenv
import json
import requests
import time
import logging

logging.basicConfig(level=logging.INFO)

load_dotenv()
app = Flask(__name__)


@app.route('/scrape-data', methods=['GET'])
def pricing():
    logging.info("Scraping data...")
    scrape()
    return "Data scraped!"


@app.route('/prices', methods=['GET'])
def get_prices():
    headers = get_requests_headers(api_key, audience)
    product_data = requests.get(f"{audience}/products", headers=headers).json()

    # with open('products.json') as file:
    #     product_data = json.load(file)

    result = get_simple_prices(product_data, low=3, high=100)
    return result


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)