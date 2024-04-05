class Stock:

    def __init__(
        self,
        name: str,
        restock_amount: int,
        restock_interval: int,
        expire_interval: int,
    ):
        self.name = name
        self.stock = restock_amount
        self.restock_amount = restock_amount
        self.restock_interval = restock_interval
        self.expire_interval = expire_interval

    def update(self, quantity_sold: int, time: int):
        # quantity sold
        self.stock -= quantity_sold
        # expire products
        if (time % self.expire_interval) == 0:
            self.expire()
        # restock event
        if (time % self.restock_interval) == 0:
            self.restock()

    def initialize(self):
        """
        Restart stock back at original stock amount.
        Assumes sale starts with one batch per item
        :return:
        """
        self.stock = self.restock_amount

    def restock(self):
        """
        inventory will be restocked with the same amount everytime.
        Batches are the same size
        :return:
        """
        self.stock += self.restock_amount

    def expire(self):
        """
        inventory will expire with the same amount everytime
        batches are the same size and expire date is per batch
        :return:
        """
        self.stock -= self.restock_amount

    def get_stock(self) -> int:
        return self.stock

    def get_name(self) -> str:
        return self.name

    def __str__(self) -> str:
        return f"{self.name}: {self.stock}"


