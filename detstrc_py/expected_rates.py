
import numpy as np
import pandas as pd
from pandas.tseries.offsets import Day, BDay

from detstrc_py.helpers import get_all_dates_table

def compute_rates(dates_for_calibration, parameters, days_to_jumps, horizons_for_calibration): 

    # compounding with daily futures
    #   parameters[0] = as_of_date rate
    #   len(parameters) - 1 = len(days_to_jumps)
    #   days_to_jump is a list of lists of dates. The dates in the sublists share the jump parameter
    #   dates_for_calibration has always calendar dates, the deltas between the dates is always 1 calendar date.
    #   it is assumed that there are 360 dates in a year.
    dates_for_calibration = dates_for_calibration.loc[dates_for_calibration["delta"] > 0]
    dates_for_calibration = dates_for_calibration.assign(rate = parameters[0])
    for dtjs, j in zip(days_to_jumps, parameters[1:]): 
        for dtj in dtjs:
            # dtj is a FOMC date. The jump on a FOMC date influences all the rates after the FOMC (it does not influence FOMC date rate), therefore > operator  
            dates_for_calibration.loc[dates_for_calibration["date"] > dtj, "rate"] += j

    dates_for_calibration["rate"] = 1 + (dates_for_calibration["rate"] * dates_for_calibration["delta"] / 360.0)
    rates = []
    for h in horizons_for_calibration: 
        this_dates = dates_for_calibration.loc[dates_for_calibration["date"] < h]
        rate = (360 / this_dates["delta"].sum()) * (this_dates["rate"].product() - 1.0)
        rates.append(rate)
    return rates

def compute_rates_linear(dates_for_calibration, parameters, days_to_jumps, horizons_for_calibration): 

    # compounding with daily futures
    #   parameters[0] = as_of_date rate
    #   len(parameters) - 1 = len(days_to_jumps)
    #   days_to_jump is a list of lists of dates. The dates in the sublists share the jump parameter
    #   dates_for_calibration has always calendar dates, the deltas between the dates is always 1 calendar date.
    #   this follows from the previous by killing the order 2 terms in rates.
    dates_for_calibration = dates_for_calibration.loc[dates_for_calibration["delta"] > 0]
    dates_for_calibration = dates_for_calibration.assign(rate = parameters[0])
    for dtjs, j in zip(days_to_jumps, parameters[1:]): 
        for dtj in dtjs: 
            dates_for_calibration.loc[dates_for_calibration["date"] > dtj, "rate"] += j
    rates = []
    for h in horizons_for_calibration: 
        this_dates = dates_for_calibration.loc[dates_for_calibration["date"] < h]
        rate = (1 / this_dates["delta"].sum()) * this_dates["rate"].sum()
        rates.append(rate)
    return rates

def predict_rates(parameters, fomc_dates, linear_estimate = False): 

    parameters["date"] = pd.to_datetime(parameters["date"])
    dates = np.array(parameters["date"].values) 
    dates.sort()
    # add dates to the date-frame until they reach last date + (horizon_month + 1 month).
    # + 1 month because we want to roll. 
    first_date = dates[0]
    last_date = dates[-1]
    last_horizon = last_date + pd.DateOffset(months = 6 + 1)
    all_dates_frame = get_all_dates_table(first_date, last_horizon)

    parameters = parameters.set_index("date")
    fomc_dates = fomc_dates.set_index("date")
    
    all_rates = []
    predicted_rates_1d = []
    predicted_rates_5d = []
    predicted_rates_10d = []
    for date in dates:
        this_parameters = parameters.loc[date].values
        this_fomcs = fomc_dates.loc[date].values
        this_jump_day_groups = [[pd.to_datetime(t)] for t in this_fomcs]
        this_horizons = [date + Day(30), date + Day(90), date + Day(180)]
        this_dates = all_dates_frame.loc[all_dates_frame["date"] >= date]
        if linear_estimate:
            this_rates = compute_rates_linear(this_dates, this_parameters, this_jump_day_groups, this_horizons)
            all_rates.append(this_rates)
        else: 
            this_rates = compute_rates(this_dates, this_parameters, this_jump_day_groups, this_horizons)
            all_rates.append(this_rates)

        this_risk_horizon = date + BDay(1)
        this_horizons = [this_risk_horizon + Day(30), this_risk_horizon + Day(90), this_risk_horizon + Day(180)]
        this_dates = all_dates_frame.loc[all_dates_frame["date"] >= this_risk_horizon]
        if linear_estimate: 
            this_rates_1d = compute_rates_linear(this_dates, this_parameters, this_jump_day_groups, this_horizons)
            predicted_rates_1d.append(this_rates_1d)                    
        else: 
            this_rates_1d = compute_rates(this_dates, this_parameters, this_jump_day_groups, this_horizons)
            predicted_rates_1d.append(this_rates_1d)

        this_risk_horizon = date + BDay(5)
        this_horizons = [this_risk_horizon + Day(30), this_risk_horizon + Day(90), this_risk_horizon + Day(180)]
        this_dates = all_dates_frame.loc[all_dates_frame["date"] >= this_risk_horizon]
        if linear_estimate: 
            this_rates_5d = compute_rates_linear(this_dates, this_parameters, this_jump_day_groups, this_horizons)
            predicted_rates_5d.append(this_rates_5d)
        else: 
            this_rates_5d = compute_rates(this_dates, this_parameters, this_jump_day_groups, this_horizons)
            predicted_rates_5d.append(this_rates_5d)

        this_risk_horizon = date + BDay(10)
        this_horizons = [this_risk_horizon + Day(30), this_risk_horizon + Day(90), this_risk_horizon + Day(180)]
        this_dates = all_dates_frame.loc[all_dates_frame["date"] >= this_risk_horizon]
        if linear_estimate: 
            this_rates_10d = compute_rates_linear(this_dates, this_parameters, this_jump_day_groups, this_horizons)
            predicted_rates_10d.append(this_rates_10d)
        else:
            this_rates_10d = compute_rates(this_dates, this_parameters, this_jump_day_groups, this_horizons)
            predicted_rates_10d.append(this_rates_10d)

        print("curve estimation: " + str(date))

    predicted_1m_rates = pd.DataFrame({
            "date":           dates,
            "rate":           [x[0] for x in all_rates],
            "rate_roll_1d":   [x[0] for x in predicted_rates_1d],
            "rate_roll_5d":   [x[0] for x in predicted_rates_5d],
            "rate_roll_10d" : [x[0] for x in predicted_rates_10d],
            "tenor" :         ["1mo" for _ in dates]}).set_index(["tenor", "date"])

    predicted_3m_rates = pd.DataFrame({
            "date":           dates, 
            "rate":           [x[1] for x in all_rates], 
            "rate_roll_1d":   [x[1] for x in predicted_rates_1d], 
            "rate_roll_5d":   [x[1] for x in predicted_rates_5d], 
            "rate_roll_10d" : [x[1] for x in predicted_rates_10d], 
            "tenor" :         ["3mo" for _ in dates]}).set_index(["tenor", "date"])

    predicted_6m_rates = pd.DataFrame({
            "date":           dates, 
            "rate":           [x[2] for x in all_rates], 
            "rate_roll_1d":   [x[2] for x in predicted_rates_1d], 
            "rate_roll_5d":   [x[2] for x in predicted_rates_5d], 
            "rate_roll_10d" : [x[2] for x in predicted_rates_10d], 
            "tenor" :         ["6mo" for _ in dates]}).set_index(["tenor", "date"])

    output_frame = pd.concat([predicted_1m_rates, predicted_3m_rates, predicted_6m_rates])
    return output_frame
