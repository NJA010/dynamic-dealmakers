# from dynamic_pricing.price import Price
from dynamic_pricing.scrape import scrape, get_requests_headers, api_key, audience, transform
from dynamic_pricing.model.price_function import price_function_sigmoid, get_simple_prices
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
    transform(endpoints=['products', 'prices'], incremental=True)
    return "Data scraped!"


@app.route('/prices', methods=['GET'])
def get_prices():

    db_client = DatabaseClient(load_config())
    query = "SELECT * FROM raw_products ORDER BY id DESC LIMIT 1"
    product_data: list = db_client.read(query)
    with open('../tests/product_data.json', 'w') as file:
        json.dump(product_data, file, default=str) 

    result = get_simple_prices(product_data[0][2], low=1, high=7)

    return json.dumps(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
