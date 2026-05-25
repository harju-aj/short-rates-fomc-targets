
import numpy as np
import pandas as pd
from pandas.tseries.offsets import Day, BDay

def get_bdays(first_date, last_date, frequency): 

    first_date_dt = pd.to_datetime(first_date)
    last_date_dt = pd.to_datetime(last_date)
    all_dates = [first_date_dt]
    running_date = first_date_dt + BDay(frequency)
    while(running_date <= last_date_dt):
        all_dates.append(running_date)
        running_date = running_date + BDay(frequency)
    all_dates = pd.DataFrame(all_dates, columns = ['date'])
    all_dates["horizon_date"] = all_dates.shift(-1)
    all_dates["time_delta"] = (all_dates["horizon_date"] - all_dates["date"]).dt.days
    return all_dates

def get_all_dates_table(first_date, last_date, bdays = False): 

    first_date = pd.to_datetime(first_date)
    last_date = pd.to_datetime(last_date)

    all_dates = [first_date]
    all_deltas = []

    d = first_date
    prev_date = first_date
    while (d < last_date): 
        if bdays: 
            d = d + BDay(1)
        else: 
            d = d + Day(1)
        all_dates.append(d)
        delta = (d - prev_date).days
        prev_date = d 
        all_deltas.append(delta)
 
    all_deltas.append(np.nan)
    output = pd.DataFrame({"date": all_dates, "delta" : all_deltas})
    output = output.sort_values("date")
    return output

def get_long_table( input, col_names ):
    output = input.copy() 
    output["date"] = pd.to_datetime(output["date"])
    output = output.set_index("date")
    output = output.stack().dropna().reset_index()
    output.columns = col_names
    return output
