import numpy as np
from dynamic_pricing.model.price_function import price_function_sigmoid
from dynamic_pricing.utils import product_index

def revenue_calc(data, x0: np.ndarray, **kwargs) -> float:
    """
    Given a price function, calculate what the revenue would have been for price data
    :param data: past price data
    :param x0: (n,1) vector with parameters. needed for scipy optimize n=len(products)*n_parameters
    :param kwargs: additional parameters of price func
    :return: the revenue
    """
    param_matrix = x0.reshape()
    data["pred_price"] = price_function_sigmoid(data.quantity.values[:, None], **kwargs)
    data["sold"] = np.where(
        (data["pred_price"] <= data["sell_price"]) & (data["pred_price"] > 0), 1, 0
    )
    pred_revenue = (data['pred_price'] * data['sold']).sum()
    return pred_revenue
