# from dynamic_pricing.price import Price
from dynamic_pricing.scrape import scrape, get_requests_headers, api_key, audience
from dynamic_pricing.model.price_function import price_function_sigmoid, get_simple_prices, get_optimized_prices
from dynamic_pricing.model.simulator import simulate
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
    stocks_data = requests.get(f"{audience}/stocks", headers=headers).json()

    # READ STOCK DATA
    try:
        db = DatabaseClient(load_config())
        params = db.read("SELECT * from params ORDER BY calc_at DESC LIMIT 10")
        params_formatted = {row[2]: [float(p) for p in row[3]] for row in params}
        result = get_optimized_prices(products_data, stocks_data, params_formatted)
        location = "optimized_prices"
    except Exception:
        result = get_simple_prices(products_data, low=1, high=7)
        location = "simple_prices"

    logging.info("Prices obtained!")

    # SAVES LOG TO DB
    output = []
    output.append([datetime.now(), json.dumps(result), location])
    db.insert_values("prices_log", output, ["simulated_at", "result", "location"])

    return json.dumps(result)


@app.route('/simulate', methods=['GET'])
def get_simulation():
    logging.info("Starting simulation...")

    simulation_data = simulate()
    total_revenue = simulation_data.pop("total_revenue")

    output = []
    output.append([datetime.now(), json.dumps(simulation_data), total_revenue])

    db = DatabaseClient(load_config())
    db.insert_values("simulation", output, ["simulated_at", "product_type", "total_revenue"])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
