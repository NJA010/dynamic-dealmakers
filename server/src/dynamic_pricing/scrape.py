import os
import requests
import google.auth.transport.requests
import google.oauth2.id_token
# from dynamic_pricing.database import load_config, DatabaseClient

api_key = os.environ.get("API_KEY")
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'pricing-prd-11719402-69eaf79e6222.json'

def get_requests_headers(api_key, audience):
    auth_req = google.auth.transport.requests.Request()
    id_token = google.oauth2.id_token.fetch_id_token(auth_req, audience)

    return {
        "X-API-KEY": api_key,
        "Authorization": f"Bearer {id_token}",
    }

audience = "https://api-4q7cwzagvq-ez.a.run.app" 
headers = get_requests_headers(api_key, audience)
# db = DatabaseClient(load_config())

# leaderboard = requests.get(f"{audience}/leaderboards", headers=headers).json()
# stocks = requests.get(f"{audience}/stocks", headers=headers).json()
prices = requests.get(f"{audience}/prices", headers=headers).json()
# products = requests.get(f"{audience}/products", headers=headers).json()
print(prices)

# Loop over every endpoint
# for endpoint in ["prices", "products", "leaderboards", "stocks"]:
#     # Get the data from the endpoint
#     data = requests.get(f"{audience}/{endpoint}", headers=headers).json()

#     # Write the data to the database
#     for d in data:
#         db.create(
#             data=d, 
#             table_name=f"raw_{endpoint}"
#         )