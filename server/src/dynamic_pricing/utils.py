import numpy as np

from dynamic_pricing.database import DatabaseClient
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import json
import os
from pathlib import Path
import datetime


products = [
    "apples-red",
    "apples-green",
    "bananas",
    "bananas-organic",
    "broccoli",
    "rice",
    "wine",
    "cheese",
    "beef",
    "avocado",
]
team_names = ["DynamicDealmakers", "GenDP", "RedAlert", "random_competitor"]

product_index = {
    "apples-red": 0,
    "apples-green": 1,
    "bananas": 2,
    "bananas-organic": 3,
    "broccoli": 4,
    "rice": 5,
    "wine": 6,
    "cheese": 7,
    "beef": 8,
    "avocado": 9,
}
index_product = {value: key for key, value in product_index.items()}
team_index = {"DynamicDealmakers": 0, "GenDP": 1, "RedAlert": 2, "random_competitor": 3}
index_team = {value: key for key, value in team_index.items()}


def get_stock(client: DatabaseClient) -> list[dict]:
    """
    Obtain current stock, lets for now just return a single stock
    across batches
    :param client:
    :return: numpy array where each index contains the stock of product i
    """
    query = "SELECT * FROM products  WHERE products.id = (select max(products.id) from products)"
    # TODO JOIN WITH STOCK ID
    with client.conn.cursor() as cur:
        data = cur.execute(query)
        data = [dict(row) for row in data]

    return data


def get_params(client: DatabaseClient) -> dict[str, np.ndarray]:
    """
    dictionary of parameters per product type
    :param client:
    :return:
    """
    query = "SELECT * FROM params"
    with client.conn.cursor() as cur:
        data = cur.execute(query)
        data = [dict(row) for row in data]

    return data


def save_params(params: dict):
    save_loc = Path(f"./runs/{datetime.datetime.now(tz=datetime.timezone.utc)}/")
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    with open(save_loc / "opt_params.json", "w") as writer:
        json.dump(params, writer, indent=4)


def get_hardcoded_sigmoid_params() -> dict[str, np.ndarray]:
    params = {
        "a": np.zeros((len(products), 1)) + 25,
        "b": np.zeros((len(products), 1)) + 10,
        "c": np.zeros((len(products), 1)) - 28,  # High for low stock
    }
    return params


class SimulatorSettings(BaseSettings):
    periods: int = Field(default=60)
    quantity_min: int = Field(default=1)
    quantity_max: int = Field(default=5)
    stock_start: int = Field(default=100)
    our_name: str
    num_teams: int