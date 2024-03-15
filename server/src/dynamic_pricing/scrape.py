import os
import requests
import google.auth.transport.requests
import google.oauth2.id_token
from dynamic_pricing.database import load_config, DatabaseClient

api_key = os.environ.get("API_KEY")
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'pricing-prd-11719402-69eaf79e6222.json'
audience = "https://api-4q7cwzagvq-ez.a.run.app" 
db = DatabaseClient(load_config())

# Function to get the headers
def get_requests_headers(api_key, audience):
    auth_req = google.auth.transport.requests.Request()
    id_token = google.oauth2.id_token.fetch_id_token(auth_req, audience)

    return {
        "X-API-KEY": api_key,
        "Authorization": f"Bearer {id_token}",
    }

# Function to scrape the data
def scrape(endpoints: list=["prices", "products", "leaderboards", "stocks"]) -> None:
    headers = get_requests_headers(api_key, audience)

    # Loop over every endpoint
    for endpoint in ["prices", "products", "leaderboards", "stocks"]:

        # Get the data from the endpoint
        data = requests.get(f"{audience}/{endpoint}", headers=headers).json()

        # Write the data to the database
        db.create(
            data=data, 
            table_name=f"raw_{endpoint}"
        )