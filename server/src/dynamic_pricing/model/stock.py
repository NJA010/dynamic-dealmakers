from jax import jit, tree_util


class SimConstant:
    def __init__(self, product, quantity):
        self.product = product
        self.quantity = quantity

    def _tree_flatten(self):
        children = (0,)
        aux_data = {
            "product": self.product,
            "quantity": self.quantity
        }
        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(**aux_data)


class Stock:

    def __init__(
        self,
        stock: int,
        name: str,
        restock_amount: int,
        restock_interval: int,
        expire_interval: int,
    ):
        self.name = name
        self.stock = stock
        self.restock_amount = restock_amount
        self.restock_interval = restock_interval
        self.expire_interval = expire_interval

    @jit
    def update_sale(self, quantity_sold: int):
        # quantity sold
        self.stock -= quantity_sold

    # @jit
    def update(self, time: int):
        # expire products
        if ((time % self.expire_interval) == 0) and (time > 0):
            self.expire()
        # restock event
        if ((time % self.restock_interval) == 0) and (time > 0):
            self.restock()

    @jit
    def initialize(self):
        """
        Restart stock back at original stock amount.
        Assumes sale starts with one batch per item
        :return:
        """
        self.stock = self.restock_amount

    @jit
    def restock(self):
        """
        inventory will be restocked with the same amount everytime.
        Batches are the same size
        :return:
        """
        self.stock += self.restock_amount

    # @jit
    def expire(self):
        """
        inventory will expire with the same amount everytime
        batches are the same size and expire date is per batch
        :return:
        """
        self.stock = max(0, self.stock - self.restock_amount)

    @jit
    def get_stock(self) -> int:
        return self.stock

    @jit
    def get_name(self) -> str:

        return self.name

    def _tree_flatten(self):
        children = (self.stock,)
        aux_data = {
            "restock_amount": self.restock_amount,
            "name": self.name,
            "restock_interval": self.restock_interval,
            "expire_interval": self.expire_interval,
        }
        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def __str__(self) -> str:
        return f"{self.name}: {self.stock}"


tree_util.register_pytree_node(Stock, Stock._tree_flatten, Stock._tree_unflatten)
tree_util.register_pytree_node(SimConstant, SimConstant._tree_flatten, SimConstant._tree_unflatten)
