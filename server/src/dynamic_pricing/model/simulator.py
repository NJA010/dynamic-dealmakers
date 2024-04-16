import datetime
import json
import os
from functools import partial

import jax
from jax.nn import relu
import numpy as np
import jax.numpy as jnp
from jax import jit, value_and_grad, random, vmap
from copy import deepcopy

# from jax.scipy.optimize import minimize
import pandas as pd
from jax._src.random import KeyArray
from scipy.optimize import minimize, LinearConstraint, differential_evolution
from typing import Generator

from dynamic_pricing.database import DatabaseClient, load_config
from dynamic_pricing.model.price_function import price_function_sigmoid
from dynamic_pricing.model.stock import Stock, SimConstant
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
    cross_join,
    save_params,
)
import logging

global PRNG_KEY

logging.basicConfig(level=logging.INFO)
logging.getLogger("jax").setLevel(logging.INFO)


@jit
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
    stock: dict[int, dict[int, Stock]], current_time: int
) -> dict[int, dict[int, Stock]]:
    # update stock for all teams
    for team, product_stock in stock.items():
        for s in product_stock.values():
            s.update(time=current_time)
    return stock


# Define a generator function to produce random numbers
def random_randint(
    rng_key: KeyArray, minval: int, maxval: int
) -> Generator[int, None, None]:
    while True:
        rng_key, subkey = random.split(rng_key)
        yield random.randint(subkey, shape=(1,), minval=minval, maxval=maxval)[0]


def get_current_stock(
    data: jnp.ndarray, stock: dict[int, dict[int, Stock]], product: int
) -> jnp.ndarray:
    """
    at a given T = t, add the current stock to the data
    :param data: data matrix
    :param stock: stock dictionary team->product->stock
    :return: data matrix with stock column at the end
    """
    stock_update = jnp.zeros(data.shape[0])
    for team in stock.keys():
        idx = jnp.where(
            (data[:, 0] == team),
            size=1,
        )[0]
        if product not in stock[team].keys():
            continue
        current_stock = stock[team][product].get_stock()
        stock_update = stock_update.at[idx].set(current_stock)

    # now set unavailale stock to inf
    stock_update = (data[:, 2] != jnp.inf) * stock_update
    data = data.at[:, 3].set(stock_update)
    return data


@jit
def distribute_units(prices, total_units):
    # Convert prices to JAX array
    prices = jnp.array(prices)

    # Calculate weights based on exponential of negative prices
    weights = prices ** (-3)

    # Normalize weights to sum up to total_units
    normalized_weights = (weights / jnp.sum(weights)) * total_units

    return normalized_weights


def approx_sold_stock(
    product_type: int,
    stock: dict[int, dict[int, Stock]],
    data: jax.Array,
    quantity: float,
    current_time: int,
) -> (list[dict], float):
    sold_quantity = distribute_units(data[:, 2], quantity)

    # update stock
    new_stock = relu(data[:, 3] - sold_quantity)
    sell_quantity = data[:, 3] - new_stock
    data = data.at[:, 3].set(new_stock)
    # save revenue and stock info
    # our team is always at zero index
    idx_us = jnp.where(data[:, 0] == 0)[0]
    if len(idx_us) > 0:
        r = jnp.nan_to_num(data[idx_us[0], 2] * sell_quantity[idx_us[0]], nan=0)
    else:
        r = 0

    valid_teams = [t for t in stock.keys() if product_type in stock[t].keys()]
    valid_teams = [t for t in valid_teams if stock[t][product_type].get_stock() > 0]
    # if not all(data[:, 0] == jnp.array(valid_teams)):
    #     raise ValueError("Team order in data matrix and stock dict do not match")
    sold_stock = []
    for team in valid_teams:
        idx = jnp.where(data[:, 0] == team)[0][0]
        sold_stock.append(
            {
                "team": team,
                "price": data[idx, 2],
                "quantity": sell_quantity[idx],
                "current_time": current_time,
            }
        )
        stock[team][product_type].update_sale(sell_quantity[idx])
    # check if some sell quantity is left
    valid_indices = data[:, 3].sum() > 1
    quantity = quantity - jnp.sum(sell_quantity)
    # Using jnp.where to conditionally call the function recursively
    # logging.debug(quantity.primal)
    while (quantity > 1) and valid_indices:
        # set price to inf is no stock left
        # p_new = jnp.where(data[:, 3] == 0, jnp.inf, data[:, 2])
        # data = data.at[:, 2].set(p_new)
        nested_sold_stock, nested_r, quantity = approx_sold_stock(
            product_type, stock, data[data[:, 3] > 0], quantity, current_time
        )
        sold_stock += nested_sold_stock
        r += nested_r

    return sold_stock, r, quantity


def obtain_sale_data(
    data: jnp.ndarray,
    stock: dict[int, dict[int, Stock]],
    quantity_dict: dict[str, int],
    current_time: int,
    product: int,
) -> (list[dict], float):
    # store revenue
    total_revenue = 0
    sold_stock = {}
    # sale quantity
    quantity = quantity_dict[index_product[product]] * 0.5
    # Filter out prices corresponding to teams with empty stock
    # Make sure its a JAX array for slicing
    if data[:, 3].sum() > 1:
        sold_stock[product], r, _ = approx_sold_stock(
            product_type=product,
            stock=stock,
            data=data[data[:, 3] > 0],
            quantity=quantity,
            current_time=current_time,
        )
        total_revenue += r
    else:
        logging.debug(f"at time: {current_time} nothing in stock")

    return sold_stock, total_revenue


# @partial(jit, static_argnums=[4])
def simulate_trades(
    x0: jnp.ndarray,
    data_us: list,
    data_competitor: list,
    original_stock: dict[int, dict[int, Stock]],
    sim_constant: SimConstant,
    **kwargs,
) -> float:
    stock = deepcopy(original_stock)
    product = sim_constant.product
    quantity_dict = sim_constant.quantity
    total_q_sold = {}
    total_q_sold[product] = 0
    # init stock
    for team in stock.keys():
        for p in stock[team].keys():
            stock[team][p].initialize()
    pred_revenue = 0.0
    # x0 = jnp.concatenate([jnp.zeros((1,)) + 50, x0])
    for current_time, (d_us_t, d_comp_t) in enumerate(zip(data_us, data_competitor)):
        d_us_t = get_current_stock(
            d_us_t,
            {t: stock[t] for t in stock.keys() if t == 0},
            product=product,
        )
        d_comp_t = get_current_stock(
            d_comp_t,
            {t: stock[t] for t in stock.keys() if t != 0},
            product=product,
        )
        # calculate price for our team, price is at idx 2
        d_us_t = d_us_t.at[:, 2].set(price_calc(x0, d_us_t, **kwargs))
        # concatenate data
        data = jnp.concatenate((d_us_t, d_comp_t))
        # set price to inf if stock is zero
        # p_new = jnp.where(data[:, 3] == 0, jnp.inf, data[:, 2])
        # data = data.at[:, 2].set(p_new)
        # for all teams determine who made the sale
        sale_data, r = obtain_sale_data(
            data,
            stock,
            quantity_dict,
            current_time,
            product,
        )
        # update revenue
        pred_revenue += r

        # logging and update stock
        for pr, sales_info in sale_data.items():
            for single_sale in sales_info:
                # update stock tracker, skip if non existed price
                # if single_sale["price"] != jnp.inf:
                #     stock[single_sale["team"]][product].update_sale(
                #         single_sale["quantity"]
                #     )
                # track and log sale
                if single_sale["team"] == 0:
                    total_q_sold[product] += single_sale["quantity"]
                logging.debug(
                    f"at time: {current_time}, {single_sale['quantity']} {index_product[product]} sold to "
                    f"{index_team[single_sale['team']]} at {single_sale['price']} "
                    f"revenue: {pred_revenue}"
                )
        # lastly restock and expire
        stock = update_stock(stock, current_time=current_time)

    logging.debug(
        f"{index_product[product]}: Total revenue = {pred_revenue} total sold = {total_q_sold[product]}"
    )
    return -pred_revenue


def run_simulation(df_price: pd.DataFrame, stock, settings: SimulatorSettings):
    # set random seed
    initial_key = random.key(42)

    # convert string type to int
    df_price = df_price.replace(product_index).replace(team_index)
    # split data in TxMxN list of matrices per product
    # split data in out data and competitor data, only ours needs to be updated
    # T = time, M = competitors, N = products
    # cross join such that for each T, N, M there's a price and stock
    all_n = df_price.competitor_name.unique()
    all_p = df_price.product_name.unique()
    all_t = df_price.scraped_at.unique()
    all_t_p = cross_join(
        [all_n, all_p, all_t], ["competitor_name", "product_name", "scraped_at"]
    )
    df_price = pd.merge(
        df_price,
        all_t_p,
        how="outer",
        on=["competitor_name", "product_name", "scraped_at"],
    )
    df_price = (
        df_price.assign(competitor_price=df_price.competitor_price.astype(float))
        .assign(competitor_name=df_price.competitor_name.astype(float))
        .assign(product_name=df_price.product_name.astype(float))
        # fill missing prices with inf
        .fillna(jnp.inf)
        # prefill stock with zero such that missing rows are skipped, but indexing works
        .assign(stock=0.0)
    )
    df_t_us = [
        jnp.array(
            df_price.loc[
                (df_price.scraped_at == t)
                & (df_price.competitor_name == team_index[settings.our_name]),
                # drop time from columns because jax does not like time
                ["competitor_name", "product_name", "competitor_price", "stock"],
            ].to_numpy()
        )
        for t in sorted(df_price.scraped_at.unique())
    ]
    df_t_comp = [
        jnp.array(
            df_price.loc[
                (df_price.scraped_at == t)
                & (df_price.competitor_name != team_index[settings.our_name]),
                # drop time from columns because jax does not like time
                ["competitor_name", "product_name", "competitor_price", "stock"],
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
        "a": jnp.array([200, 200, 300, 100, 200, 100, 40, 40, 40, 40]).reshape(10, 1),
        "b": jnp.array([5, 5, 5, 5, 5, 5, 10, 10, 10, 10]).reshape(10, 1) * 2,
        "c": jnp.zeros((10, 1)) - 20,  # High for low stock
    }
    opt_params = {}
    total_revenue = 0
    # do simulation per product
    for prod, i in product_index.items():
        logging.info(
            f"Running optimization for: {prod} with time length: {len(df_t_us)}"
        )
        x0 = jnp.concatenate([params["a"][i], params["b"][i], params["c"][i]])

        # a, b >0
        bounds = ((1, 1000), (1, 100), (-50, 50))
        # r = simulate_trades(
        #     x0,
        #     [d[d[:, 1] == i] for d in df_t_us],
        #     [d[d[:, 1] == i] for d in df_t_comp],
        #     stock_mapped,
        #     randstock_gen,
        # )

        # logging.info(r)
        obj_and_grad = value_and_grad(simulate_trades)
        sim_constant = SimConstant(product=i, quantity=settings.quantity)
        res = minimize(
            obj_and_grad,
            x0,
            args=(
                [d[d[:, 1] == i] for d in df_t_us],
                [d[d[:, 1] == i] for d in df_t_comp],
                stock_mapped,
                sim_constant,
            ),
            method="L-BFGS-B",
            bounds=bounds,
            # seed=42,
            # disp=True
            options={"disp": True, "gtol": 1e-4},
            jac=True,
        )
        total_revenue += -float(res.fun)
        logging.info(
            f"Optimal parameters for {prod} is {res.x} with {-res.fun} revenue"
        )
        opt_params[prod] = [float(p) for p in res.x]
    opt_params["total_revenue"] = total_revenue
    logging.info(f"Total revenue is {total_revenue}")
    return opt_params


if __name__ == "__main__":
    settings = SimulatorSettings(
        quantity_min=1,
        quantity_max=5,
        our_name="DynamicDealmakers",
        num_teams=len(team_names),
    )
    # GENERATE DATA
    # Define number of products and time periods
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
    # df_prices = pd.DataFrame(
    #     data,
    #     columns=["competitor_name", "product_name", "scraped_at", "competitor_price"],
    # )

    # DATABASE
    db_client = DatabaseClient(load_config())
    df_prices = get_prices(db_client, interval="2 hour")

    # READ LOCAl JSON
    # df_prices = pd.read_json("./prices.json")

    # agg prices per batch for simplicity
    df_prices = (
        df_prices.groupby(["competitor_name", "scraped_at", "product_name"])
        .competitor_price.mean()
        .reset_index()
    )
    # create stock objects
    stock = {team: {} for team in team_names}
    _ = {
        stock[row[1]["competitor_name"]].update(
            {
                row[1]["product_name"]: Stock(
                    name=row[1]["product_name"],
                    stock=settings.stock_start[row[1]["product_name"]],
                    restock_amount=settings.stock_start[row[1]["product_name"]],
                    restock_interval=settings.restock_interval[row[1]["product_name"]],
                    expire_interval=settings.expire_interval[row[1]["product_name"]],
                )
            }
        )
        for row in df_prices[["competitor_name", "product_name"]]
        .drop_duplicates()
        .iterrows()
    }
    simulation_data = run_simulation(df_prices, stock, settings)
    save_params(simulation_data)
    # simulation_data = json.load(open('./params.json'))
    # opt_list = unwrap_params(opt)
    # db_client.insert_values(
    #     table_name="params",
    #     values=opt_list,
    #     column_names=["calc_at", "product_name", "opt_params"],
    # )
    total_revenue = simulation_data.pop("total_revenue")

    output = []
    ts = datetime.datetime.now()
    max_id = db_client.read("SELECT MAX(id) from simulation")[0][0]
    if max_id is None:
        max_id = 1
    else:
        max_id += 1
    for obj in simulation_data:
        output.append([max_id, ts, obj, simulation_data[obj], total_revenue])

    db_client.insert_values(
        "simulation",
        output,
        ["id", "simulated_at", "product_name", "params", "total_revenue"],
    )
    logging.info("Sucessfully added parameters to the params table")
