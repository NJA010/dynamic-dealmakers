import os
import logging
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
def scrape(endpoints: list[str] = ENDPOINTS) -> None:
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
        db.create(
            data=data.json(), 
            table_name=f"raw_{endpoint}"
        )
        
        logging.info("Data written to the database!")



def transform(
    endpoints: list[str] = ENDPOINTS,
    incremental: bool = True
) -> None:

    for endpoint in endpoints:
        max_id = db.read_max_id(endpoint) if incremental else 0
        max_id = max_id if max_id is not None else 0
        
        path = Path(__file__).parent / "sql" / f"{endpoint}.sql"
        with open(path) as file:
            environment = jinja2.Environment()
            template = environment.from_string(file.read())
            query = template.render(max_id=max_id)
            db.query_no_return(query)

        logging.info(f"Raw table tranformation for {endpoint} complete")
        

if __name__ == "__main__":
    import json
    
    headers = get_requests_headers(api_key, audience)
    data = requests.get(f"{audience}/products", headers=headers).json()

    # Write the data to a JSON file
    with open("data.json", "w") as file:
        json.dump(data, file)

