import numpy as np
import jax.numpy as jnp
from jax import jit, value_and_grad
import pandas as pd
from scipy.optimize import minimize, LinearConstraint

from dynamic_pricing.model.price_function import price_function_sigmoid
from dynamic_pricing.utils import product_index
import logging


logging.basicConfig(level=logging.INFO)


def price_calc(x0: np.ndarray, data, **kwargs) -> float:
    """
    Given a price function, calculate what the revenue would have been for price data
    :param data: past price data
    :param x0: (n,1) vector with parameters. needed for scipy optimize, format is like
    (a,b,c,a,b,c,a,b,c...) each parameter per product type
    :param kwargs: additional parameters of price func
    :return: the revenue
    """
    x0_matrix = x0.reshape(len(np.unique(data[:, 1])), 3)

    p = price_function_sigmoid(data[:, -1], *x0_matrix.T, **kwargs)

    return p


def update_stock(
    sale_data: dict, stock: dict[str, dict[str, int]]
) -> dict[str, dict[str, int]]:

    # update stock
    for product, sales_info in sale_data.items():
        stock[sales_info["team"]][product] -= sales_info["quantity"]
    return stock


def get_current_stock(data: np.ndarray, stock: dict[str, dict[str, int]]) -> np.ndarray:
    """
    at a given T = t, add the current stock to the data
    :param data: data matrix
    :param stock: stock dictionary team->product->stock
    :return: data matrix with stock column at the end
    """
    stock_update = np.zeros(data.shape[0])
    for team in stock.keys():
        for product in stock[team].keys():
            mask = (data[:, 0] == team) & (data[:, 1] == product)
            if stock[team][product] > 0:
                stock_update[mask] = stock[team][product]

    data = np.c_[data, stock_update]
    return data


def obtain_sale_data(data: np.ndarray, stock: dict[str, dict[str, int]]):
    # Extract relevant columns from the array
    teams = data[:, 0]
    product_types = data[:, 1]
    sell_prices = data[:, 3]

    # Get unique product types
    unique_product_types = np.unique(product_types)

    # Create dictionaries to store the lowest sell price and corresponding team
    sale_data = {}
    # store revenue
    r = 0
    # Iterate over each unique product type
    for product_type in unique_product_types:
        # Find indices where the product type matches
        indices = np.where(product_types == product_type)[0]
        # sale quantity
        quantity = np.random.randint(1, 5)
        # Filter out prices corresponding to teams with stock larger than zero
        valid_indices = [
            idx for idx in indices if stock[teams[idx]][product_type] >= quantity
        ]
        prices_for_product_type = sell_prices[valid_indices]
        if len(prices_for_product_type) > 0:
            # Find the index of the minimum sell price
            min_price_index = np.argmin(prices_for_product_type)
            # Use this index to get the corresponding team ID
            team_id_with_lowest_price = teams[valid_indices[min_price_index]]
            # Store the lowest price, corresponding team ID, quantity
            lowest_price = prices_for_product_type[min_price_index]

            sale_data[product_type] = {
                "team": team_id_with_lowest_price,
                "price": lowest_price,
                "quantity": quantity,
            }
            # store revenue
            if team_id_with_lowest_price == "Team_1":
                r += quantity * lowest_price
    return sale_data, r


def simulate_trades(
    x0: np.ndarray,
    data_us: list,
    data_competitor: list,
    stock: dict[str, dict[str, int]],
    **kwargs,
):
    pred_revenue = 0

    for d_us_t, d_comp_t in zip(data_us, data_competitor):
        d_us_t = get_current_stock(d_us_t, stock)
        d_comp_t = get_current_stock(d_comp_t, stock)
        # calculate price for our team
        d_us_t[:, 3] = price_calc(x0, d_us_t, **kwargs)
        # for all teams determine who made the sale
        sale_data, r = obtain_sale_data(np.concatenate((d_us_t, d_comp_t)), stock)
        # update revenue
        pred_revenue += r

        stock = update_stock(sale_data, stock)

    return -pred_revenue


def run_simulation(df_price, stock):
    our_name = "Team_1"
    # split data in TxMxN list of matrices per product
    # split data in out data and competitor data, only ours needs to be updated
    # T = time, M = competitors, N = products
    df_t_us = [
        df_price[(df_price.time == t) & (df_price.team == our_name)].to_numpy()
        for t in sorted(df_price.time.unique())
    ]
    df_t_comp = [
        df_price[(df_price.time == t) & (df_price.team != our_name)].to_numpy()
        for t in sorted(df_price.time.unique())
    ]
    # initialize parameters
    params = {
        "a": np.zeros((10, 1)) + 50,
        "b": np.zeros((10, 1)) + 10,
        "c": np.zeros((10, 1)) - 20,  # High for low stock
    }
    opt_params = {}
    # do simulation per product
    for prod, i in product_index.items():
        logging.info(f"Running optimization for: {prod}")
        x0 = np.concatenate(
            [params["a"][i], params["b"][i], params["c"][i]]
        )

        # a, b >0
        bounds = ((1, np.inf), (1, 20), (-50, 50))
        # simulate_trades(x0, df_t_us, df_t_comp, stock)
        res = minimize(
            simulate_trades,
            x0,
            args=(
                [d[d[:, 1] == prod] for d in df_t_us],
                [d[d[:, 1] == prod] for d in df_t_comp],
                stock,
            ),
            method="L-BFGS-B",
            bounds=bounds,
            options={"disp": True, "gtol": 1e-3, "eps": 1e-15},
        )
        logging.info(f"Optimal parameters for {prod} is {res.x} with {-res.fun} revenue")
        opt_params[prod] = res.x
    return opt_params


if __name__ == "__main__":
    # Define number of products and time periods
    num_products = 10
    num_periods = 60
    num_teams = 4

    # Generate random data for selling price and quantity for each team
    data = []
    periods = pd.date_range(start="2024-01-01", periods=num_periods, freq="min")

    for team in range(num_teams):
        np.random.seed(team)  # Setting seed to ensure same data for each team
        selling_prices = np.random.uniform(0, 2, size=(num_products, num_periods))
        team_name = f"Team_{team+1}"
        for j, product_name in enumerate(product_index.keys()):
            for i, period in enumerate(periods):
                data.append(
                    [
                        team_name,
                        product_name,
                        period,
                        selling_prices[j, i],
                    ]
                )

    # Create DataFrame
    df = pd.DataFrame(data, columns=["team", "product_type", "time", "sell_price"])
    stock = {c: {p: 100 for p in product_index.keys()} for c in df.team.unique()}

    opt = run_simulation(df, stock)
