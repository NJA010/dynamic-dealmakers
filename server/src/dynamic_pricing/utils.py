import time
from typing import Any

import numpy as np
import pandas as pd
import pytz

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
team_names = [
    "DynamicDealmakers",
    "GenDP",
    "RedAlert",
    "random_competitor",
    "ThePRIceIsRight",
]

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
team_index = {
    "DynamicDealmakers": 0,
    "GenDP": 1,
    "RedAlert": 2,
    "random_competitor": 3,
    "ThePRIceIsRight": 4,
}
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

def get_prices(client: DatabaseClient, interval: str = "1 hour"):
    query = f"""
    select 
    *
    from prices 
    where 1=1 
    and scraped_at >= (select max(scraped_at) from prices) - interval '{interval}' 
    """
    data, description = client.read_df(query)
    df = pd.DataFrame(data, columns=[desc[0] for desc in description])
    return df


def save_params(params: dict):
    save_loc = Path(f"./runs/{datetime.datetime.now(tz=datetime.timezone.utc)}/")
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    with open(save_loc / "opt_params.json", "w") as writer:
        json.dump(params, writer, indent=4)


def unwrap_params(params: dict[dict[dict[Any]]]) -> list[list[Any]]:
    amsterdam_tz = pytz.timezone("Europe/Amsterdam")
    ts = datetime.datetime.now(amsterdam_tz)
    output = []
    for key, value in params.items():
        output.append([ts, key, value])
    return output


def get_hardcoded_sigmoid_params() -> dict[str, np.ndarray]:
    params = {
        "a": np.zeros((len(products), 1)) + 25,
        "b": np.zeros((len(products), 1)) + 10,
        "c": np.zeros((len(products), 1)) - 28,  # High for low stock
    }
    return params


class SimulatorSettings(BaseSettings):
    periods: int = Field(default=600)
    quantity_min: int = Field(default=1)
    quantity_max: int = Field(default=5)
    quantity: dict = {
        "apples-red": 55,
        "apples-green": 42,
        "bananas": 80,
        "bananas-organic": 21,
        "broccoli": 42,
        "rice": 42,
        "wine": 16,
        "cheese": 11,
        "beef": 22,
        "avocado": 12,
    }
    stock_start: dict = {
        "apples-red": 150,
        "apples-green": 100,
        "bananas": 200,
        "bananas-organic": 50,
        "broccoli": 100,
        "rice": 50,
        "wine": 20,
        "cheese": 30,
        "beef": 30,
        "avocado": 20,
    }
    # restock time in minutes
    restock_interval: dict = {
        "apples-red": 11,
        "apples-green": 11,
        "bananas": 11,
        "bananas-organic": 11,
        "broccoli": 11,
        "rice": 20,
        "wine": 20,
        "cheese": 11,
        "beef": 11,
        "avocado": 11,
    }
    expire_interval: dict = {
        "apples-red": 60,
        "apples-green": 60,
        "bananas": 60,
        "bananas-organic": 60,
        "broccoli": 60,
        "rice": 60*24,
        "wine": 60*24,
        "cheese": 60,
        "beef": 60,
        "avocado": 30,
    }
    our_name: str
    num_teams: int
