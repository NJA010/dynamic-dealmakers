import os
import logging
from datetime import datetime, timedelta
import pytz
from typing import Any, Optional
import pytz
from dataclasses import dataclass
import json

import requests

import google.auth.transport.requests
import google.oauth2.id_token

from dynamic_pricing.database import load_config, DatabaseClient

logging.basicConfig(level=logging.INFO)

api_key = os.environ.get("TF_VAR_api_key")
audience = "https://api-4q7cwzagvq-ez.a.run.app"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'pricing-prd-11719402-69eaf79e6222.json'
db = DatabaseClient(load_config())

@dataclass
class TableConfig:
    name: str
    cutoff_interval_hours: int
    time_column: str

ENDPOINTS = ["prices", "products", "leaderboards", "stocks"]
TABLE_CONFIGS = [
    TableConfig('prices', 24, 'scraped_at'),
    TableConfig('products', 24, 'scraped_at'),
    TableConfig('leaderboards', 24*7, 'scraped_at'),
    TableConfig('stocks', 24, 'scraped_at'),
    TableConfig('prices_log', 24, 'scraped_at'),
]


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

        last_response = db.read(f"SELECT response FROM last_endpoint_responses WHERE endpoint = '{endpoint}'")[0][0]

        if last_response == data.json():
            logging.info(f"Response for endpoint '{endpoint}' same as last run, skipping databse insert.")
            continue

        amsterdam_tz = pytz.timezone('Europe/Amsterdam')
        ts = datetime.now(amsterdam_tz)

        db.query_no_return(f"""
            update last_endpoint_responses
            set
                scraped_at = '{ts}',
                response = '{json.dumps(data.json())}'
            where endpoint = '{endpoint}'
        """)
        
        # Write the data to the database
        logging.info(f"Writing data from {endpoint} to the database...")

        try:
            match endpoint:
                case "prices":
                    max_id = get_max_id(db, 'prices', 'WHERE id < 1000000000')
                    output = unwrap_prices(data.json(), ts, max_id)
                    db.insert_values(endpoint, output, ['id', 'scraped_at', 'product_name', 'batch_name', 'competitor_name', 'competitor_price'])
                case "products":
                    max_id = get_max_id(db, 'products', 'WHERE id < 1000000000')
                    output = unwrap_products(data.json(), ts, max_id)
                    db.insert_values(endpoint, output, ['id', 'scraped_at', 'product_name', 'batch_key', 'batch_id', 'batch_expiry'])
                case "stocks":
                    max_id = get_max_id(db, 'stocks', 'WHERE id < 1000000000')
                    output = unwrap_stocks(data.json(), ts, max_id)
                    for row in output:
                        try:
                            last = db.read('SELECT stock_amount '
                                    'FROM stocks '
                                    f'WHERE batch_id={row[2]} '
                                    'ORDER BY id DESC LIMIT 1')[0][0]

                            current_and_last_is_positive = int(row[3]) > 0 and int(row[4]) > 0

                            if not current_and_last_is_positive:
                                continue

                            row.append(last)
                            row.append(int(row[4]) - int(row[3]))
                        except IndexError:
                            row.append(None)
                            row.append(None)
                    db.insert_values(endpoint, output, ['id', 'scraped_at', 'batch_id', 'stock_amount', 'prev_stock_amount', 'sold_stock'])
                case "leaderboards":
                    max_id = get_max_id(db, 'leaderboards', 'WHERE id < 1000000000')
                    output = unwrap_leaderboards(data.json(), ts, max_id)
                    db.insert_values(endpoint, output, ['id', 'scraped_at', 'team_name', 'score'])
                case _:
                    continue
        except KeyError:
            logging.error(f"Could not unwrap json data for {endpoint}. Data: {data.json()}")
            return

        logging.info("Data written to the database!")

    clean_old_records(TABLE_CONFIGS)





def unwrap_products(response_data: dict[str, dict[str, Any]], ts: datetime, id: int) -> list[list[Any]]:
    output = []
    for product_name, product_value in response_data.items():
        for batch_key, batch_values in product_value['products'].items():
            output.append([id, ts, product_name, batch_key, batch_values['id'], datetime.fromisoformat(batch_values['sell_by'])])

    return output


def unwrap_stocks(response_data: dict[str, dict[str, Any]], ts: datetime, id: int) -> list[list[Any]]:
    output = []
    for value in response_data.values():
        for batch_id, stock_amount in value.items():
            output.append([id, ts, batch_id, stock_amount])

    return output


def unwrap_prices(response_data: dict[str, dict[str, dict[str, Any]]], ts: datetime, id: int) -> list[list[Any]]:
    output = []
    for product, value in response_data.items():
        for batch_id, competitor_data in value.items():
            for competitor_id, price in competitor_data.items():
                output.append([id, ts, product, batch_id, competitor_id, price])

    return output


def unwrap_leaderboards(response_data: dict[str, str], ts: datetime, id: int) -> list[list[Any]]:
    output = []
    for team_name, score in response_data.items():
        output.append([id, ts, team_name, score])

    return output


def get_max_id(connection, table_name: str, where: str) -> int:
    max_id = connection.read(f"SELECT MAX(id) from {table_name} {where}")[0][0]
    if max_id is None:
        max_id = 1
    else:
        max_id += 1
    return max_id



def clean_old_records(config: list[TableConfig]) -> None:
    for conf in config:
        stale_cutoff = datetime.now(pytz.timezone('Europe/Amsterdam')) - timedelta(hours=conf.cutoff_interval_hours)
        db.query_no_return(f"DELETE FROM {conf.name} WHERE scraped_at < '{str(stale_cutoff)}'")
        logging.info(f"Stale data for '{conf.name}' has been removed.")


if __name__ == "__main__":
    scrape()

