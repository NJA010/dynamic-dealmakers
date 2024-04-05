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
from dynamic_pricing.model.stock import Stock
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
    save_params,
)
import logging

global PRNG_KEY

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("jax").setLevel(logging.INFO)


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
    stock: dict[int, dict[int, Stock]],
    current_time: int
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
    data: jnp.ndarray, stock: dict[int, dict[int, Stock]]
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
            current_stock = stock[team][product].get_stock()
            if current_stock > 0:
                stock_update = stock_update.at[idx].set(current_stock)

    data = jnp.c_[data, stock_update]
    return data


def get_sold_stock(
    product_type: int,
    stock: dict[int, dict[int, Stock]],
    valid_prices: jnp.ndarray,
    valid_teams: jnp.ndarray,
    quantity: int,
    valid_indices: jnp.ndarray,
    current_time: int,
) -> (list[dict], float):
    sold_stock = []
    # sort indices based on prices
    sorted_indices = valid_indices[valid_prices.argsort()]
    # per valid team/price, obtain current stock and sell remainder to rest
    i = 0
    # store revenue
    r = 0
    team_sell = int(valid_teams[sorted_indices[i]])
    while quantity > 0 and i< sorted_indices.size:
        # obtain index/team corresponding to cheapest price
        idx = sorted_indices[i]
        # only sell once per team
        if (team_sell == int(valid_teams[idx])) and i>0:
            i+=1
            continue
        else:
            team_sell = int(valid_teams[idx])
        price_sell = float(valid_prices[idx].primal)
        # check how much this team can sell and save
        current_stock = stock[team_sell][int(product_type)].get_stock()
        sale = min(current_stock, quantity)
        sold_stock.append(
            {
                "team": team_sell,
                "price": price_sell,
                "quantity": sale,
                "current_time": current_time,
            }
        )
        # update stock
        stock[team_sell][int(product_type)].update_sale(sale)
        # store revenue if our team has best valid price
        if team_sell == 0:
            r = r + sale * price_sell
        # in case 0<stock<quantity, sell remainder to next team
        quantity -= sale
        i += 1
    return sold_stock, r


def obtain_sale_data(
    data: jnp.ndarray,
    stock: dict[int, dict[int, Stock]],
    settings: SimulatorSettings,
    current_time: int,
) -> (list[dict], float):
    # Extract relevant columns from the array
    teams = data[:, 0]
    product_types = data[:, 1]
    sell_prices = data[:, 2]

    # Get unique product types
    unique_product_types = jnp.unique(product_types)

    # store revenue
    total_revenue = []
    sold_stock = {}
    # Iterate over each unique product type
    for product_type in unique_product_types:
        # Find indices where the product type matches
        indices = jnp.where(product_types == product_type)[0]
        # sale quantity
        quantity = settings.quantity[index_product[int(product_type)]]
        # Filter out prices corresponding to teams with empty stock
        # Make sure its a JAX array for slicing
        valid_indices = jnp.array(
            [
                idx
                for idx in indices
                if stock[int(teams[idx])][int(product_type)].get_stock() > 0
            ]
        )

        if len(valid_indices) > 0:
            sold_stock[int(product_type)], r = get_sold_stock(
                product_type=int(product_type),
                stock=stock,
                valid_prices=sell_prices[valid_indices],
                valid_teams=teams[valid_indices],
                quantity=quantity,
                valid_indices=valid_indices,
                current_time=current_time,
            )
            total_revenue.append(r)
        else:
            logging.debug(f"Time: {current_time}, No valid indices all out of stock")
    return sold_stock, sum(total_revenue)


def simulate_trades(
    x0: jnp.ndarray,
    data_us: list,
    data_competitor: list,
    original_stock: dict[int, dict[int, Stock]],
    settings: SimulatorSettings,
    **kwargs,
) -> float:

    stock = deepcopy(original_stock)
    # init stock
    for team in stock.keys():
        for product in stock[team].keys():
            stock[team][product].initialize()
    pred_revenue = 0.0
    # x0 = jnp.concatenate([jnp.zeros((1,)) + 50, x0])
    for current_time, (d_us_t, d_comp_t) in enumerate(zip(data_us, data_competitor)):
        d_us_t = get_current_stock(d_us_t, stock)
        d_comp_t = get_current_stock(d_comp_t, stock)
        # calculate price for our team, price is at idx 2
        d_us_t = d_us_t.at[:, 2].set(price_calc(x0, d_us_t, **kwargs))
        # for all teams determine who made the sale
        sale_data, r = obtain_sale_data(
            jnp.concatenate((d_us_t, d_comp_t)), stock, settings, current_time
        )
        # update revenue
        pred_revenue += r

        stock = update_stock(stock, current_time=current_time)
        for pr, sales_info in sale_data.items():
            for single_sale in sales_info:
                logging.debug(
                    f"at time: {current_time}, {single_sale['quantity']} {index_product[pr]} sold to "
                    f"{index_team[single_sale['team']]} at {single_sale['price']}"
                    f"revenue: {single_sale['quantity']*single_sale['price']}"
                )
    return -pred_revenue


def run_simulation(df_price, stock, settings: SimulatorSettings):
    # set random seed
    initial_key = random.key(42)

    # convert string type to int
    df_price = df_price.replace(product_index).replace(team_index)
    # split data in TxMxN list of matrices per product
    # split data in out data and competitor data, only ours needs to be updated
    # T = time, M = competitors, N = products
    df_price = (
        df_price.assign(competitor_price=df_price.competitor_price.astype(float))
        .assign(competitor_name=df_price.competitor_name.astype(float))
        .assign(product_name=df_price.product_name.astype(float))
    )
    df_t_us = [
        jnp.array(
            df_price.loc[
                (df_price.scraped_at == t)
                & (df_price.competitor_name == team_index[settings.our_name]),
                # drop time from columns because jax does not like time
                ["competitor_name", "product_name", "competitor_price"],
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
    total_revenue = 0
    # do simulation per product
    for prod, i in product_index.items():
        logging.info(f"Running optimization for: {prod} with time length: {len(df_t_us)}")
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
                settings,
            ),
            method="L-BFGS-B",
            bounds=bounds,
            # seed=42,
            # disp=True
            options={"disp": True, "gtol": 1e-12},
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
    df_prices = get_prices(db_client, interval='1 hour')

    # READ LOCAl JSON
    # df_prices = pd.read_json("./prices.json")
    # create stock objects
    stock = {
        c: {
            p: Stock(
                name=p,
                restock_amount=settings.stock_start[p],
                restock_interval=settings.restock_interval[p],
                expire_interval=settings.expire_interval[p],
            )
            for p in product_index.keys()
        }
        for c in team_names
    }
    opt = run_simulation(df_prices, stock, settings)
    save_params(opt)
    # opt = json.load(open('./tests/opt_params.json'))
    # opt_list = unwrap_params(opt)
    # db_client.insert_values(
    #     table_name="params",
    #     values=opt_list,
    #     column_names=["calc_at", "product_name", "opt_params"],
    # )
    logging.info("Sucessfully added parameters to the params table")
