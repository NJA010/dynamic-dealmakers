import numpy as np

from dynamic_pricing.database import DatabaseClient
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


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
team_index = {
    "Team_1": 0,
    "Team_2": 1,
    "Team_3": 2,
    "Team_4": 3
}


def get_stock(client: DatabaseClient) -> np.ndarray:
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


def get_hardcoded_sigmoid_params() -> dict[str, np.ndarray]:
    params = {"a": np.zeros((len(products), 1)) + 25,
              "b": np.zeros((len(products), 1)) + 10,
              "c": np.zeros((len(products), 1)) - 28 # High for low stock
              }
    return params


class SimulatorSettings(BaseSettings):
    periods: int = Field(default=60)
    quantity_min: int = Field(default=1)
    quantity_max: int = Field(default=5)

