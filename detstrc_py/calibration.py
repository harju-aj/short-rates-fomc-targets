import numpy as np
import pandas as pd
import datetime, calendar
from pandas.tseries.offsets import Day, BDay
from dateutil.relativedelta import relativedelta
from scipy import optimize

from detstrc_py.helpers import get_all_dates_table, get_long_table

one_month_in_years = 1.0 / 12.0
two_month_in_years = 1.0 / 6.0
three_months_in_years = 0.25
six_months_in_years = 0.5

def future_prices_error(jump_means):

    prices = compute_future_prices(rates_for_calibration, days_in_reference_months_for_calibration, fomc_months_indicator_for_calibration, 
                                   jump_means, days_to_fomc_for_calibration)
    
    diff = [ (x - y)**2 for x,y in zip(prices, market_prices_for_calibration)]
    error = np.sum(np.array(diff))
    error = np.sqrt(error)
    return error 

def rates_error(parameters):

    rates = compute_rates(dates_for_calibration, parameters, n_days_to_fomc_for_calibration, horizons_for_calibration)
    diff = [ (x - y)**2 for x,y in zip(rates, rates_for_calibration)]
    error = np.sum(np.array(diff))
    error = np.sqrt(error)
    return error


def compute_future_prices(historical_rates, dates_each_month, jump_months_indicator, jump_means, days_to_jump):
    prices = []
    rates_sum = np.sum(historical_rates)
    # historical dates have been filled to include all dates in the calendar:
    n_historical_rate_dates = len(historical_rates)
    fomc_index = 0
    r_0 = historical_rates[-1]
    dynamic_mean_rate = r_0

    # 1st reference month:
    if jump_months_indicator[0] == 1:
        n_days_to_jump = days_to_jump[0] - n_historical_rate_dates
        n_days_after_jump = dates_each_month[0] - days_to_jump[0]
        rates_sum = rates_sum + r_0 * n_days_to_jump
        dynamic_mean_rate = r_0 + jump_means[fomc_index]
        rates_sum = rates_sum + n_days_after_jump * dynamic_mean_rate
        fomc_index = fomc_index + 1
    else:
        n_days_after_jump = dates_each_month[0] - n_historical_rate_dates
        rates_sum = rates_sum + n_days_after_jump * r_0
    average_rate = 100 * rates_sum / dates_each_month[0]
    prices.append(100 - average_rate)

    # remaining reference months:
    for N, I, dtj in zip(dates_each_month[1:], jump_months_indicator[1:], days_to_jump[1:]):
        if I == 1:
            rates_sum = dtj * dynamic_mean_rate
            n_days_after_jump = N - dtj
            dynamic_mean_rate = dynamic_mean_rate + jump_means[fomc_index]
            rates_sum = rates_sum + n_days_after_jump * dynamic_mean_rate
            fomc_index = fomc_index + 1
        else:
            rates_sum = dynamic_mean_rate * N

        average_rate = 100 * rates_sum / N
        prices.append(100 - average_rate)

    return prices

def compute_rates(dates_for_calibration, parameters, days_to_jumps, horizons_for_calibration): 

    # parameters[0] = as_of_date rate 
    # len(parameters) - 1 = len(days_to_jumps)
    # days_to_jump is a list of lists of dates. The dates in the sublists share the jump parameter 
    #  
    # The following drops the last date from the dates_from_calibration array silently. 
    # The resulting array has 180 days = 6 monts for calibration - this is OK.
    dates_for_calibration = dates_for_calibration.loc[dates_for_calibration["delta"] > 0]
    dates_for_calibration = dates_for_calibration.assign(rate = parameters[0])
    for dtjs, j in zip(days_to_jumps, parameters[1:]): 
        for dtj in dtjs: 
            dates_for_calibration.loc[dates_for_calibration["date"] > dtj, "rate"] += j

    dates_for_calibration["rate"] = 1 + (dates_for_calibration["rate"] * dates_for_calibration["delta"] / 360.0)
    rates = []
    for h in horizons_for_calibration: 
        this_dates = dates_for_calibration.loc[dates_for_calibration["date"] < h]
        rate = (360 / this_dates["delta"].sum()) * (this_dates["rate"].product() - 1.0)
        rates.append(rate)
    return rates

# effr_rates for each calendar date!
def infer_parameters_from_futures(date, prices, FOMC_dates, historical_rates_frame, date_column, rate_column ):

    global rates_for_calibration
    global days_in_reference_months_for_calibration
    global fomc_months_indicator_for_calibration
    global days_to_fomc_for_calibration 
    global market_prices_for_calibration

    n_fomc_dates = len(FOMC_dates)
    n_reference_months = len(prices) 

    date = pd.to_datetime(date)
    year = date.year 
    month = date.month

    FOMC_dates = [pd.to_datetime(x) for x in FOMC_dates]
    
    first_day_this_month = datetime.datetime(date.year, date.month, 1)

    # The current month rates (those that are available on the valuation date) will be needed in the pricing formula of the current month future:
    rates_this_month = historical_rates_frame.loc[(historical_rates_frame[date_column] >= first_day_this_month) & (historical_rates_frame[date_column] <= date)]
    rates_this_month = rates_this_month[rate_column].values
    
    # Find the (year, month) pairs for the reference contracts: 
    reference_months = [(date.year, date.month)]
    first_day_ref_month = first_day_this_month
    for i in np.arange(n_reference_months - 1):
        first_day_ref_month = first_day_ref_month + relativedelta(months=1)
        reference_months.append((first_day_ref_month.year, first_day_ref_month.month))
    
    # Find the days in reference months:
    days_in_reference_months = []
    for y, m in reference_months:
        n_days_this_month = calendar.monthrange(y, m)[1]
        days_in_reference_months.append(n_days_this_month)

    # For each reference month, get a FOMC month indicator, and count the days to FOMC. 
    fomc_months_indicator = np.zeros(n_reference_months)
    days_to_fomc = np.zeros(n_reference_months)
    for fomc_date in FOMC_dates: 
        fomc_year = fomc_date.year
        fomc_month = fomc_date.month
        fomc_day = fomc_date.day
        i = 0
        for year, month in reference_months:
            if year == fomc_year and month == fomc_month: 
                fomc_months_indicator[i] = 1
                days_to_fomc[i] = fomc_day
                break
            i = i + 1
    
    rates_for_calibration = rates_this_month
    days_in_reference_months_for_calibration = days_in_reference_months
    fomc_months_indicator_for_calibration = fomc_months_indicator
    days_to_fomc_for_calibration = days_to_fomc
    market_prices_for_calibration = prices
    # discount_factors_for_calibration = discount_factors
    
    jump_means_prior = [0.002] * n_fomc_dates
    result = optimize.minimize(future_prices_error, jump_means_prior, method='Nelder-Mead', tol=0.000001)
    return result.x

def infer_parameters_from_curve(this_date, this_rates, this_fomc_dates, this_effr_rate, n_jumps_from_curve):

    global this_effr_for_calibration
    global n_days_to_fomc_for_calibration
    global rates_for_calibration
    global horizons_for_calibration
    global dates_for_calibration 

    this_date_dt = pd.to_datetime(this_date)
    n_days_to_fomcs = [ (pd.to_datetime(fomc_date) - this_date_dt).days for fomc_date in this_fomc_dates ]
    horizon_1_months = this_date_dt + Day(30)
    horizon_3_months = this_date_dt + Day(90)
    horizon_6_months = this_date_dt + Day(180)
    days_to_horizon_3m = (horizon_3_months - this_date_dt).days # = 90 
    days_to_horizon_6m = (horizon_6_months - this_date_dt).days # = 180

    horizons_for_calibration = [horizon_1_months, horizon_3_months, horizon_6_months]
    rates_for_calibration = this_rates

    n_days_to_fomc_for_calibration, parameters_prior = None, None

    if n_jumps_from_curve == 1: 
        fomcs_before_6m = [pd.to_datetime(d) for n,d in zip(n_days_to_fomcs, this_fomc_dates) if n < days_to_horizon_6m]
        n_days_to_fomc_for_calibration = [fomcs_before_6m]
    else: 
        fomcs_before_3m = [pd.to_datetime(d) for n,d in zip(n_days_to_fomcs, this_fomc_dates) if n < days_to_horizon_3m]
        fomcs_3m_to_6m = [pd.to_datetime(d) for n,d in zip(n_days_to_fomcs, this_fomc_dates) if (n < days_to_horizon_6m and n >= days_to_horizon_3m)]
        n_days_to_fomc_for_calibration = [fomcs_before_3m, fomcs_3m_to_6m]
    dates_for_calibration = get_all_dates_table(this_date_dt, horizon_6_months)

    jump_means_prior = [0.002] * n_jumps_from_curve
    parameters_prior = [this_effr_rate] + jump_means_prior
    result = optimize.minimize(rates_error, parameters_prior, method='Nelder-Mead', tol=0.000001)
    
    if n_jumps_from_curve == 1:
        output = (result.x[0], [result.x[1]] * len(this_fomc_dates))
    else: 
        n_near_fomcs = len(fomcs_before_3m)
        n_later_fomc = len(this_fomc_dates) - n_near_fomcs
        output = (result.x[0], ([result.x[1]] * n_near_fomcs) + ([result.x[2]] * n_later_fomc))
    return output 


# effr_rates for each calendar date!
def calibrate_parameters(effr_rates, fomc_dates, fomc_dates_alt, futures_prices, rates, training_horizon_shift):
    
    n_jumps_from_curve = 1
    number_of_reference_months_used_for_calibration = 7 + training_horizon_shift
    K = number_of_reference_months_used_for_calibration # (short name)
    add_rate_deltas = True # The calibration algorithm also produces forward looking delta(effr)'s. 

    futures_prices = get_futures_prices_matrix(futures_prices, K)

    # The model is calibrated on those dates that has futures prices for reference months: 0, 1, ..., number_of_reference_months_used_for_calibration
    dates_with_futures = futures_prices[["date"]]
    rates = rates.merge(dates_with_futures)
    rates = rates.sort_values("date")
    
    n_days = dates_with_futures.shape[0]
    days_to_fomc_collection = []
    days_to_alt_fomc_collection = []
    cum_rate_delta_collection = []
    parameters_collection = []
    fomc_dates_collection = []
    fomc_alt_dates_collection = []
    parameters_from_rates_collection = []
    overnight_rates = []
    calibrated_dates = []
    error_dates = []
    errors = 0
    for i in np.arange(n_days):
        this_date = futures_prices.iloc[i,0]
        try: 
            this_fomc_dates = get_fomc_dates_in_window(this_date, fomc_dates, K)
            this_fomc_alt_dates = get_fomc_dates_in_window(this_date, fomc_dates_alt, K)
            this_fomc_alt_dates = this_fomc_alt_dates[0:len(this_fomc_dates)]
            this_days_to_fomc = [(pd.to_datetime(t) - pd.to_datetime(this_date)).days for t in this_fomc_dates]
            this_days_to_fomc_alt = [(pd.to_datetime(t) - pd.to_datetime(this_date)).days for t in this_fomc_alt_dates]
            this_fomc_dates_plus_1bd =  [pd.to_datetime(t) +  BDay(1) for t in this_fomc_dates]
            this_effr_rate = effr_rates.loc[effr_rates["date"] == this_date].iloc[0,1]    

            this_rates = rates.loc[rates['date'] == this_date].iloc[0, 1:].values
            this_futures_prices = futures_prices.iloc[i,1:].values 

            if add_rate_deltas: 
                rate_fomc_plus_1 = [effr_rates.loc[pd.to_datetime(effr_rates["date"]) == t].iloc[0,1] for t in this_fomc_dates_plus_1bd]
                rate_delta = [p - this_effr_rate for p in rate_fomc_plus_1]
                cum_rate_delta_collection.append(rate_delta)
  
            # The calibration needs the futures prices for the reference months 0, 1, ... , K
            parameters = infer_parameters_from_futures(this_date , this_futures_prices, this_fomc_dates, effr_rates, "date", "effr" )
            overnight_rate, parameters_from_rates = infer_parameters_from_curve(this_date, this_rates, this_fomc_dates, this_effr_rate, n_jumps_from_curve)
            #
            days_to_fomc_collection.append(this_days_to_fomc)
            days_to_alt_fomc_collection.append(this_days_to_fomc_alt)
            
            parameters_collection.append(parameters)
            fomc_dates_collection.append(this_fomc_dates)
            fomc_alt_dates_collection.append(this_fomc_alt_dates)
            parameters_from_rates_collection.append(parameters_from_rates)
            overnight_rates.append(overnight_rate)
            calibrated_dates.append(this_date)
            print("calibration success: " + str(this_date))
        except:
            error_dates.append(this_date)
            errors = errors+1
            print("calibration fails: " + str(this_date))

    print("fails: " + str(errors))

    fomc_dates_frame = pd.DataFrame(fomc_dates_collection)
    fomc_dates_frame["date"] = pd.to_datetime(calibrated_dates)
    fomc_dates_frame = get_long_table( fomc_dates_frame, ["date", "fomc", "fomc_date"] )

    alt_fomc_dates_frame = pd.DataFrame(fomc_alt_dates_collection)
    alt_fomc_dates_frame["date"] = pd.to_datetime(calibrated_dates)
    alt_fomc_dates_frame = get_long_table( alt_fomc_dates_frame, ["date", "fomc", "alt_fomc_date"] )

    days_to_fomc_frame = pd.DataFrame(days_to_fomc_collection)
    days_to_fomc_frame["date"] = pd.to_datetime(calibrated_dates)
    days_to_fomc_frame = get_long_table( days_to_fomc_frame, ["date", "fomc", "days_to_fomc"] )

    alt_days_to_fomc_frame = pd.DataFrame(days_to_alt_fomc_collection)
    alt_days_to_fomc_frame["date"] = pd.to_datetime(calibrated_dates)
    alt_days_to_fomc_frame = get_long_table( alt_days_to_fomc_frame, ["date", "fomc", "alt_days_to_fomc"] )

    if add_rate_deltas: 
        effr_delta_frame = pd.DataFrame(cum_rate_delta_collection)
        effr_delta_frame["date"] = pd.to_datetime(calibrated_dates)
        effr_delta_frame = get_long_table( effr_delta_frame, ["date", "fomc", "effr_delta"] )

    parameters_frame = pd.DataFrame(parameters_collection)
    jump_sum_frame = parameters_frame.copy()
    jump_columns = jump_sum_frame.columns 
    for x, y in zip(jump_columns[0:-1], jump_columns[1:]):
        jump_sum_frame[y] = jump_sum_frame[y] + jump_sum_frame[x]

    jump_sum_frame["date"] = pd.to_datetime(calibrated_dates)
    jump_sum_frame = get_long_table( jump_sum_frame, ["date", "fomc", "jump_sum"] )

    parameters_frame["date"] = pd.to_datetime(calibrated_dates)
    parameters_frame = get_long_table( parameters_frame, ["date", "fomc", "jump"] )

    overnight_rates_frame = pd.DataFrame(overnight_rates)
    overnight_rates_frame.columns = ["overnight_rate"]
    overnight_rates_frame["date"] = pd.to_datetime(calibrated_dates)

    parameters_from_rates_frame = pd.DataFrame(parameters_from_rates_collection)
    jump_sum_frame2 = parameters_from_rates_frame.copy()
    jump_columns2 = jump_sum_frame2.columns 
    for x, y in zip(jump_columns2[0:-1], jump_columns2[1:]):
        jump_sum_frame2[y] = jump_sum_frame2[y] + jump_sum_frame2[x] 

    jump_sum_frame2["date"] = pd.to_datetime(calibrated_dates)
    jump_sum_frame2 = get_long_table( jump_sum_frame2, ["date", "fomc", "jump_sum_bench"] )

    parameters_from_rates_frame["date"] = pd.to_datetime(calibrated_dates)
    parameters_from_rates_frame = get_long_table( parameters_from_rates_frame, ["date", "fomc", "jump_bench"] )

    output = fomc_dates_frame.merge(alt_fomc_dates_frame, on = ["date", "fomc"])
    output = output.merge(days_to_fomc_frame, on = ["date", "fomc"])
    output = output.merge(alt_days_to_fomc_frame, on = ["date", "fomc"])
    output = output.merge(effr_rates, on = ["date"])
    if add_rate_deltas:
        output = output.merge(effr_delta_frame, on =  ["date", "fomc"])
    output = output.merge(parameters_frame, on =  ["date", "fomc"])
    output = output.merge(jump_sum_frame, on =  ["date", "fomc"])
    output = output.merge(overnight_rates_frame, on =  ["date"])
    output = output.merge(parameters_from_rates_frame, on =  ["date", "fomc"])
    output = output.merge(jump_sum_frame2, on = ["date", "fomc"])
    return output

def get_fomc_dates_in_window(date, fomc_dates_frame, window_in_months): 
    start_date = pd.to_datetime(date)
    end_date = start_date + pd.DateOffset(months=window_in_months)
    fomc_dates = sorted(fomc_dates_frame.loc[(fomc_dates_frame["date"] >= start_date) & (fomc_dates_frame["date"] < end_date)]["date"].values)
    fomc_dates = [str(x)[0:10] for x in fomc_dates] # str(x)[0:10] is a hack that gets the YYYY-MM-DD string off datetime.
    return fomc_dates

def get_futures_prices_matrix(futures_prices, K, drop_nas = True):

    futures_prices_matrix = futures_prices.copy()
    futures_prices_matrix['year'] = futures_prices_matrix['year'].apply(lambda x: int(x))
    futures_prices_matrix['month'] = futures_prices_matrix['month'].apply(lambda x: int(x))
    futures_prices_matrix['quote_year'] = futures_prices_matrix["date"].apply(lambda x: x.year)
    futures_prices_matrix['quote_month'] = futures_prices_matrix["date"].apply(lambda x: x.month)
    years = futures_prices_matrix['year'].values
    quote_years = futures_prices_matrix['quote_year'].values
    months = futures_prices_matrix['month'].values
    quote_months = futures_prices_matrix['quote_month'].values
    contract_ints = []
    for y, qy, m, qm in zip(years, quote_years, months, quote_months): 
        this_contract_int = 12 * (y - qy) + (m - qm)
        contract_ints.append(this_contract_int)
    futures_prices_matrix['contract_int'] = contract_ints
    futures_prices_matrix = futures_prices_matrix.loc[(futures_prices_matrix['contract_int'] >= 0) & (futures_prices_matrix['contract_int'] <= K)]
    futures_prices_matrix = futures_prices_matrix.pivot(index = 'date', columns = 'contract_int')['price'].reset_index()
    if drop_nas: 
        futures_prices_matrix = futures_prices_matrix.dropna()
    return futures_prices_matrix


