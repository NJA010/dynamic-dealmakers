# from dynamic_pricing.price import Price
from dynamic_pricing.scrape import scrape, get_requests_headers, api_key, audience
from dynamic_pricing.model.price_function import price_function_sigmoid, get_simple_prices, get_optimized_prices
from dynamic_pricing.utils import get_stock, get_params
from dynamic_pricing.database import DatabaseClient, load_config
from flask import Flask
from dotenv import load_dotenv
import json
import requests
import time
import logging
from datetime import datetime

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
    logging.info("Getting prices...")
    headers = get_requests_headers(api_key, audience)
    products_data = requests.get(f"{audience}/products", headers=headers).json()
    # stocks_data = requests.get(f"{audience}/stocks", headers=headers).json()

    result = get_simple_prices(products_data, low=1, high=7)
    logging.info("Prices obtained!")
    # READ STOCK DATA
    # stock = get_stock()
    # params = requests.get(f"{audience}/params", headers=headers).json()
    # result = get_optimized_prices(products_data, stocks_data, params)
    # READ PARAMETERS
    # OBTAIN PRICE
    return json.dumps(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
