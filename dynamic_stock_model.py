import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm

def compute_outflow(inflow, lifetime, sd):
    """
    Computes the outflow based on the inflow, lifetime and standard deviation.
    """
    # Initialize variables
    year_complete = np.arange(1900, 1901)
    outflow = np.repeat(0, len(year_complete))

    # Iterate over years
    for k in range(1901, 2021):
        # Compute outflow for each year
        outflow_list = inflow.iloc[0:len(year_complete)] * norm.pdf(k - year_complete, lifetime.iloc[0:len(year_complete)],  sd.iloc[0:len(year_complete)])

        # Sum up outflow for the year
        outflow_sum = np.sum(outflow_list)

        # Append the outflow sum to the outflow list
        outflow = np.append(outflow, outflow_sum)

        # Append the current year to the year_complete list
        year_complete = np.append(year_complete, k)

    # Convert the outflow list to a pandas Series and return
    return pd.Series(outflow)

def compute_stock(inflow, outflow):
    """
    Computes the stock based on the inflow and outflow.
    """
    stock = inflow.sub(outflow).cumsum()
    return stock