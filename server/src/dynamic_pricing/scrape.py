import os
import logging
from datetime import datetime
from typing import Any, Optional
import time

import requests

import google.auth.transport.requests
import google.oauth2.id_token

from dynamic_pricing.database import load_config, DatabaseClient

logging.basicConfig(level=logging.INFO)

api_key = os.environ.get("TF_VAR_api_key")
audience = "https://api-4q7cwzagvq-ez.a.run.app"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'pricing-prd-11719402-69eaf79e6222.json'
db = DatabaseClient(load_config())

ENDPOINTS = ["prices", "products", "leaderboards", "stocks"]


# Function to get the headers
def get_requests_headers(api_key, audience):
    auth_req = google.auth.transport.requests.Request()
    id_token = google.oauth2.id_token.fetch_id_token(auth_req, audience)

    return {
        "X-API-KEY": api_key,
        "Authorization": f"Bearer {id_token}",
    }


# Function to scrape the data
def scrape(endpoints: Optional[list[str]] = None) -> None:
    if endpoints is None:
        endpoints = ENDPOINTS

    headers = get_requests_headers(api_key, audience)

    # Loop over every endpoint
    for endpoint in endpoints:
        logging.info(f"\nScraping data from {endpoint}...")

        # Get the data from the endpoint
        data = requests.get(f"{audience}/{endpoint}", headers=headers)

        if data.status_code != 200:
            logging.error(
                f"\nFailed to get data from {endpoint}.\
                \n\tStatus code: {data.status_code}\
                \n\tResponse: {data.json()}"
            )
            continue

        logging.info(f"\tStatus code: {data.status_code}")
        
        # Write the data to the database
        logging.info(f"Writing data from {endpoint} to the database...")
        try:
            match endpoint:
                case "prices":
                    output = unwrap_prices(data.json())
                    db.insert_values(endpoint, output, ['id', 'scraped_at', 'product_name', 'batch_name', 'competitor_name', 'competitor_price'])
                case "products":
                    output = unwrap_products(data.json())
                    db.insert_values(endpoint, output, ['id', 'scraped_at', 'product_name', 'batch_key', 'batch_id', 'batch_expiry'])
                case "stocks":
                    output = unwrap_stocks(data.json())
                    for row in output:
                        try:
                            last = db.read('SELECT stock_amount '
                                    'FROM stocks '
                                    f'WHERE batch_id={row[2]} '
                                    'ORDER BY id DESC LIMIT 1')[0][0]
                            row.append(last)
                            row.append(int(row[4]) - int(row[3]))
                        except IndexError:
                            row.append(None)
                            row.append(None)
                    db.insert_values(endpoint, output, ['id', 'scraped_at', 'batch_id', 'stock_amount', 'prev_stock_amount', 'sold_stock'])
                case _:
                    continue
        except KeyError:
            logging.error(f"Could not unwrap json data for {endpoint}. Data: {data.json()}")
            return

        logging.info("Data written to the database!")


def unwrap_products(response_data: dict[dict[Any]]) -> list[list[Any]]:
    now = datetime.now()
    id = int(time.time())
    output = []
    for product_name, product_value in response_data.items():
        for batch_key, batch_values in product_value['products'].items():
            output.append([id, now, product_name, batch_key, batch_values['id'], datetime.fromisoformat(batch_values['sell_by'])])

    return output


def unwrap_stocks(response_data: dict[dict[Any]]) -> list[list[Any]]:
    now = datetime.now()
    id = int(time.time())
    output = []
    for key, value in response_data.items():
        for batch_id, stock_amount in value.items():
            output.append([id, now, batch_id, stock_amount])

    return output


def unwrap_prices(response_data: dict[dict[dict[Any]]]) -> list[list[Any]]:
    now = datetime.now()
    id = int(time.time())
    output = []
    for product, value in response_data.items():
        for batch_id, competitor_data in value.items():
            for competitor_id, price in competitor_data.items():
                output.append([id, now, product, batch_id, competitor_id, price])

    return output


if __name__ == "__main__":
    import json

    scrape()
    # data = requests.get(f"{audience}/products", headers=headers).json()


