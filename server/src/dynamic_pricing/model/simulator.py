import json

import numpy as np
import jax.numpy as jnp
from jax import jit, value_and_grad, random
from copy import deepcopy

# from jax.scipy.optimize import minimize
import pandas as pd
from jax._src.random import KeyArray
from scipy.optimize import minimize, LinearConstraint, differential_evolution
from typing import Generator

from dynamic_pricing.database import DatabaseClient, load_config
from dynamic_pricing.model.price_function import price_function_sigmoid
from dynamic_pricing.utils import (
    product_index,
    team_index,
    SimulatorSettings,
    index_team,
    index_product,
    team_names,
    products,
    get_prices,
    unwrap_params,
    save_params
)
import logging

global PRNG_KEY

logging.basicConfig(level=logging.INFO)


def price_calc(x0: jnp.ndarray, data, **kwargs) -> float:
    """
    Given a price function, calculate what the revenue would have been for price data
    :param data: past price data
    :param x0: (n,1) vector with parameters. needed for scipy optimize, format is like
    (a,b,c,a,b,c,a,b,c...) each parameter per product type
    :param kwargs: additional parameters of price func
    :return: the revenue
    """
    # x0_matrix = x0.reshape(jnp.size(data[:, 1]), 3)

    p = price_function_sigmoid(data[:, -1], *x0, **kwargs)

    return p


def update_stock(
    sale_data: dict, stock: dict[int, dict[int, int]]
) -> dict[int, dict[int, int]]:

    # update stock
    for product, sales_info in sale_data.items():
        stock[sales_info["team"]][product] -= sales_info["quantity"]
    return stock


# Define a generator function to produce random numbers
def random_randint(
    rng_key: KeyArray, minval: int, maxval: int
) -> Generator[int, None, None]:
    while True:
        rng_key, subkey = random.split(rng_key)
        yield random.randint(subkey, shape=(1,), minval=minval, maxval=maxval)[0]


def get_current_stock(
    data: jnp.ndarray, stock: dict[int, dict[int, int]]
) -> jnp.ndarray:
    """
    at a given T = t, add the current stock to the data
    :param data: data matrix
    :param stock: stock dictionary team->product->stock
    :return: data matrix with stock column at the end
    """
    stock_update = jnp.zeros(data.shape[0])
    for team in stock.keys():
        for product in stock[team].keys():
            idx = jnp.where((data[:, 0] == team) & (data[:, 1] == product))[0]
            if stock[team][product] > 0:
                stock_update = stock_update.at[idx].set(stock[team][product])

    data = jnp.c_[data, stock_update]
    return data


def obtain_sale_data(
    data: jnp.ndarray,
    stock: dict[int, dict[int, int]],
    randstock_gen: Generator[int, None, None],
) -> (dict, float):
    # Extract relevant columns from the array
    teams = data[:, 0]
    product_types = data[:, 1]
    sell_prices = data[:, 2]

    # Get unique product types
    unique_product_types = jnp.unique(product_types)

    # Create dictionaries to store the lowest sell price and corresponding team
    sale_data = {}
    # store revenue
    r = 0
    # Iterate over each unique product type
    for product_type in unique_product_types:
        # Find indices where the product type matches
        indices = jnp.where(product_types == product_type)[0]
        # sale quantity
        quantity = next(randstock_gen)
        # Filter out prices corresponding to teams with stock smaller than sell quantity
        # Make sure its a JAX array for slicing
        valid_indices = jnp.array(
            [
                idx
                for idx in indices
                if stock[int(teams[idx])][int(product_type)] >= quantity
            ]
        )
        if len(valid_indices) > 0:
            prices_for_product_type = sell_prices[valid_indices]
            # Find the index of the minimum sell price
            min_price_index = jnp.argmin(prices_for_product_type)
            # Use this index to get the corresponding team ID
            team_id_with_lowest_price = teams[valid_indices[min_price_index]]
            # Store the lowest price, corresponding team ID, quantity
            lowest_price = prices_for_product_type[min_price_index]

            sale_data[int(product_type)] = {
                "team": int(team_id_with_lowest_price),
                "price": lowest_price,
                "quantity": int(quantity),
            }
            # store revenue if our team has best valid price
            if team_id_with_lowest_price == 0:
                r = r + quantity * lowest_price
    return sale_data, jnp.sum(r)


def simulate_trades(
    x0: jnp.ndarray,
    data_us: list,
    data_competitor: list,
    original_stock: dict[int, dict[int, int]],
    randstock_gen: Generator[int, None, None],
    **kwargs,
) -> float:

    stock = deepcopy(original_stock)
    pred_revenue = 0.0
    # x0 = jnp.concatenate([jnp.zeros((1,)) + 50, x0])
    for d_us_t, d_comp_t in zip(data_us, data_competitor):
        d_us_t = get_current_stock(d_us_t, stock)
        d_comp_t = get_current_stock(d_comp_t, stock)
        # calculate price for our team, price is at idx 2
        d_us_t = d_us_t.at[:, 2].set(price_calc(x0, d_us_t, **kwargs))
        # for all teams determine who made the sale
        sale_data, r = obtain_sale_data(
            jnp.concatenate((d_us_t, d_comp_t)), stock, randstock_gen
        )
        # update revenue
        pred_revenue += r

        stock = update_stock(sale_data, stock)
        for pr, sale in sale_data.items():
            logging.debug(
                f"{sale['quantity']} {index_product[pr]} sold to {index_team[sale['team']]} at {sale['price']}"
            )
    return -pred_revenue


def run_simulation(df_price, stock, settings: SimulatorSettings):
    # set random seed
    initial_key = random.key(42)
    randstock_gen = random_randint(
        initial_key,
        minval=settings.quantity_min,
        maxval=settings.quantity_max,
    )
    # convert string type to int
    df_price = df_price.replace(product_index).replace(team_index)
    # split data in TxMxN list of matrices per product
    # split data in out data and competitor data, only ours needs to be updated
    # T = time, M = competitors, N = products
    df_price = (
        df_price
        .assign(competitor_price=df_price.competitor_price.astype(float))
        .assign(competitor_name=df_price.competitor_name.astype(float))
        .assign(product_name=df_price.product_name.astype(float))

    )
    df_t_us = [
        jnp.array(
            df_price.loc[
                (df_price.scraped_at == t) & (df_price.competitor_name == team_index[settings.our_name]),
                # drop time from columns because jax does not like time
                ["competitor_name", "product_name", "competitor_price"],
            ].to_numpy()
        )
        for t in sorted(df_price.scraped_at.unique())
    ]
    df_t_comp = [
        jnp.array(
            df_price.loc[
                (df_price.scraped_at == t) & (df_price.competitor_name != team_index[settings.our_name]),
                # drop time from columns because jax does not like time
                ["competitor_name", "product_name", "competitor_price"],
            ].to_numpy()
        )
        for t in sorted(df_price.scraped_at.unique())
    ]
    # convert team and product names to ints
    stock_mapped = {
        team_index[team]: {
            product_index[product]: s for product, s in stock[team].items()
        }
        for team in stock.keys()
    }
    # initialize parameters
    params = {
        "a": jnp.zeros((10, 1)) + 50,
        "b": jnp.zeros((10, 1)) + 10,
        "c": jnp.zeros((10, 1)) - 20,  # High for low stock
    }
    opt_params = {}
    # do simulation per product
    for prod, i in product_index.items():
        logging.info(f"Running optimization for: {prod}")
        x0 = jnp.concatenate([params["a"][i], params["b"][i], params["c"][i]])

        # a, b >0
        bounds = ((1, 200), (1, 50), (-50, 50))
        # r = simulate_trades(
        #     x0,
        #     [d[d[:, 1] == i] for d in df_t_us],
        #     [d[d[:, 1] == i] for d in df_t_comp],
        #     stock_mapped,
        #     randstock_gen,
        # )

        # logging.info(r)
        obj_and_grad = value_and_grad(simulate_trades)
        res = minimize(
            obj_and_grad,
            x0,
            args=(
                [d[d[:, 1] == i] for d in df_t_us],
                [d[d[:, 1] == i] for d in df_t_comp],
                stock_mapped,
                randstock_gen,
            ),
            method="L-BFGS-B",
            bounds=bounds,
            # seed=42,
            # disp=True
            options={"disp": True, "gtol": 1e-12},
            jac=True,
        )
        logging.info(
            f"Optimal parameters for {prod} is {res.x} with {-res.fun} revenue"
        )
        opt_params[prod] = [float(p) for p in res.x]
    return opt_params


if __name__ == "__main__":
    settings = SimulatorSettings(
        quantity_min=1, quantity_max=10, our_name="DynamicDealmakers", num_teams=len(team_names)
    )
    # # Define number of products and time periods
    # num_products = len(products)
    # num_periods = settings.periods
    # num_teams = settings.num_teams
    #
    # # Generate random data for selling price and quantity for each team
    # data = []
    # periods = pd.date_range(start="2024-01-01", periods=num_periods, freq="min")
    #
    # for team in range(num_teams):
    #     np.random.seed(team)  # Setting seed to ensure same data for each team
    #     selling_prices = np.random.uniform(0, 10, size=(num_products, num_periods))
    #     team_name = team_names[team]
    #     for j, product_name in enumerate(product_index.keys()):
    #         for i, period in enumerate(periods):
    #             data.append(
    #                 [
    #                     team_name,
    #                     product_name,
    #                     period,
    #                     selling_prices[j, i],
    #                 ]
    #             )
    #
    # # Create DataFrame
    # df = pd.DataFrame(data, columns=["competitor_name", "product_name", "scraped_at", "competitor_price"])

    db_client = DatabaseClient(load_config())
    df_prices = get_prices(db_client)

    stock = {
        c: {p: settings.stock_start for p in product_index.keys()}
        for c in df_prices.competitor_name.unique()
    }
    opt = run_simulation(df_prices, stock, settings)
    save_params(opt)
    # opt = json.load(open('./tests/opt_params.json'))
    opt_list = unwrap_params(opt)
    db_client.insert_values(table_name="params", values=opt_list, column_names=["calc_at", "product_name", "opt_params"])
    logging.info("Sucessfully added parameters to the params table")