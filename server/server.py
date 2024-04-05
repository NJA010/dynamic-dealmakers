# from dynamic_pricing.price import Price
from dynamic_pricing.model.stock import Stock
from dynamic_pricing.scrape import scrape, get_requests_headers, api_key, audience
from dynamic_pricing.model.price_function import get_simple_prices, get_optimized_prices
from dynamic_pricing.model.simulator import run_simulation
from dynamic_pricing.utils import (
    get_stock,
    get_params,
    SimulatorSettings,
    team_names,
    product_index,
)
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

    db = DatabaseClient(load_config())
    # READ STOCK DATA
    try:
        params = db.read("SELECT * from simulation ORDER BY calc_at DESC LIMIT 10")
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
    settings = SimulatorSettings(
        quantity_min=1,
        quantity_max=5,
        our_name="DynamicDealmakers",
        num_teams=len(team_names),
    )
    # set starting stock levels
    stock = {
        c: {
            p: Stock(
                name=p,
                restock_amount=settings.stock_start[p],
                restock_interval=settings.restock_interval[p],
                expire_interval=settings.expire_interval[p],
            )
            for p in product_index.keys()
        }
        for c in team_names
    }
    # obtain prices data
    db_client = DatabaseClient(load_config())
    df_prices = get_prices(db_client, interval='5 hour')
    # run simulation
    simulation_data = run_simulation(df_prices, stock, settings)
    total_revenue = simulation_data.pop("total_revenue")

    output = []
    ts = datetime.now()
    max_id = db_client.read("SELECT MAX(id) from simulation")[0][0]
    if max_id is None:
        max_id = 1
    else:
        max_id += 1
    for obj in simulation_data:
        output.append([max_id, ts, obj, simulation_data[obj], total_revenue])

    db_client.insert_values("simulation", output, ["id", "simulated_at", "product_name", "params", "total_revenue"])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
