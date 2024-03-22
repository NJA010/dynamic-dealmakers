import numpy as np


def revenue_calc(data, price_func, **kwargs) -> float:
    """
    Given a price function, calculate what the revenue would have been for price data
    :param data: past price data
    :param price_func: function to calculate the price
    :param kwargs: additional parameters of price func
    :return: the revenue
    """
    data["pred_price"] = price_func(data.quantity.values[:, None], **kwargs)
    data["sold"] = np.where(
        (data["pred_price"] <= data["sell_price"]) & (data["pred_price"] > 0), 1, 0
    )
    pred_revenue = (data['pred_price'] * data['sold']).sum()
    return pred_revenue
