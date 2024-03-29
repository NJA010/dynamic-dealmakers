import os
import logging
from datetime import datetime
from typing import Any, Optional

import requests

import google.auth.transport.requests
import google.oauth2.id_token
import jinja2
from pathlib import Path

from dynamic_pricing.database import load_config, DatabaseClient

logging.basicConfig(level=logging.INFO)

api_key = os.environ.get("TF_VAR_api_key")
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'pricing-prd-11719402-69eaf79e6222.json'
audience = "https://api-4q7cwzagvq-ez.a.run.app" 
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
        data = requests.get(f"{audience}/{endpoint}", headers=headers).json()

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
        match endpoint:
            case "prices":
                output = unwrap_prices(data)
            case "products":
                output = unwrap_products(data)
            case "stocks":
                output = unwrap_stocks(data)
            case _:
                continue

        db.insert_values(endpoint, output)
        logging.info("Data written to the database!")


def unwrap_products(response_data: dict[dict[Any]]) -> list[list[Any]]:
    now = datetime.now()
    output = []
    for product_name, product_value in response_data.items():
        for batch_key, batch_values in product_value['products'].items():
            output.append([now, product_name, batch_key, batch_values['id'], datetime.fromisoformat(batch_values['sell_by'])])

    return output


def unwrap_stocks(response_data: dict[dict[Any]]) -> list[list[Any]]:
    now = datetime.now()
    output = []
    for key, value in response_data.items():
        for batch_id, stock_amount in value.items():
            output.append([now, key, batch_id, stock_amount])

    return output


def unwrap_prices(response_data: dict[dict[dict[Any]]]) -> list[list[Any]]:
    now = datetime.now()
    output = []
    for product, value in response_data.items():
        for batch_id, competitor_data in value.items():
            for competitor_id, price in competitor_data.items():
                output.append([now, product, batch_id, competitor_id, price])

    return output

if __name__ == "__main__":
    import json
    
    headers = get_requests_headers(api_key, audience)
    data = requests.get(f"{audience}/products", headers=headers).json()

    # Write the data to a JSON file
    with open("data.json", "w") as file:
        json.dump(data, file)

