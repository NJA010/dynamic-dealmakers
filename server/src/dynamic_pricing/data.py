from pydantic import BaseModel
import datetime


class Prices(BaseModel):
    """
    This is where the results are which are served to the /get prices request
    """
    serve_time: datetime.time
    product_type: str
    product_uuid: str
    batch_id: str
    price: float


class Stock(BaseModel):
    """
    Combines raw stock and products.
    For every timestamp, what is our stock?
    """
    time: datetime.time
    product_type: str
    product_uuid: str
    sell_by: datetime.time
    batch_id: str
    stock: int


class SoldProducts(BaseModel):
    """
    Based on the Change in stock and prices,
    obtain sold quantity with price.
    For every timestamp, how much did we sell and for what price?
    """
    time: datetime.time
    product_type: str
    product_uuid: str
    batch_id: str
    quantity: int
    sell_price: int


class AllProducts(BaseModel):
    """
    Contains all products that were sold by all teams
    Based on raw prices table
    """
    time: datetime.time
    product_type: str
    sell_price: float
    quantity: int
