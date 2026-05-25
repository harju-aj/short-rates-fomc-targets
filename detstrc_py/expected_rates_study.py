
import numpy as np
import pandas as pd 
import statsmodels.api as sm
from pandas.tseries.offsets import BDay
from scipy.stats import binomtest

short_horizon = 1
one_month_in_years = 1.0 / 12.0
two_month_in_years = 1.0 / 6.0
three_months_in_years = 0.25
six_months_in_years = 0.5

def get_rate_deltas(input, list_of_calendars, horizons):
    
    rates = input.copy()
    rates.index = rates['date']
    output_frames = []
    for x, y in zip(list_of_calendars, horizons): 
        deltas_1mo = []
        deltas_3mo = []
        deltas_6mo = []
        calendar_dates = x["date"].values
        calendar_dates = calendar_dates.copy()
        calendar_dates.sort()
        for xx_plus_one, xx in zip(calendar_dates[1:], calendar_dates[0:-1]):
            deltas_1mo.append(rates.loc[xx_plus_one]["1mo"] - rates.loc[xx]["1mo"])
            deltas_3mo.append(rates.loc[xx_plus_one]["3mo"] - rates.loc[xx]["3mo"])
            deltas_6mo.append(rates.loc[xx_plus_one]["6mo"] - rates.loc[xx]["6mo"])
            0 == 0
        deltas_1mo = pd.DataFrame({"date": calendar_dates[0:-1], "delta_" + str(y) + "d": deltas_1mo})
        deltas_1mo["tenor"] = "1mo"
        deltas_3mo = pd.DataFrame({"date": calendar_dates[0:-1], "delta_" + str(y) + "d": deltas_3mo})
        deltas_3mo["tenor"] = "3mo"
        deltas_6mo = pd.DataFrame({"date": calendar_dates[0:-1], "delta_" + str(y) + "d": deltas_6mo})
        deltas_6mo["tenor"] = "6mo"
        deltas_this_tenor = pd.concat([deltas_1mo, deltas_3mo, deltas_6mo])
        output_frames.append(deltas_this_tenor)
    return output_frames[0], output_frames[1], output_frames[2]

def get_fomc_indicators(dates, fomc_dates): 
    
    fomc_dates["date"] = pd.to_datetime(fomc_dates["fomc-date"], format='%Y%m%d')
    fomc_dates_list = fomc_dates["date"].values
    all_dates = dates["date"].values 
    fomc_1d_indicator = []
    fomc_5d_indicator = []
    fomc_10d_indicator = []
    for x in all_dates: 
        if x in fomc_dates_list: 
            fomc_1d_indicator.append(1)
        else: 
            fomc_1d_indicator.append(0)
    for x in all_dates: 
        if (x in fomc_dates_list) or x + BDay(1) in fomc_dates_list or x + BDay(2) in fomc_dates_list or x + BDay(3) in fomc_dates_list or x + BDay(4) in fomc_dates_list : 
            fomc_5d_indicator.append(1)
        else: 
            fomc_5d_indicator.append(0)
    for x in all_dates: 
        if (x in fomc_dates_list) or x + BDay(1) in fomc_dates_list or x + BDay(2) in fomc_dates_list or x + BDay(3) in fomc_dates_list or x + BDay(4) in fomc_dates_list or \
           x + BDay(5) in fomc_dates_list or x + BDay(6) in fomc_dates_list or x + BDay(7) in fomc_dates_list or x + BDay(8) in fomc_dates_list or x + BDay(9) in fomc_dates_list : 
            fomc_10d_indicator.append(1)
        else: 
            fomc_10d_indicator.append(0)
    output = pd.DataFrame({"date" : all_dates, "fomc_1d_window" : fomc_1d_indicator, "fomc_5d_window" : fomc_5d_indicator, "fomc_10d_window" : fomc_10d_indicator})
    output["no_fomc"] = np.where(output["fomc_10d_window"] == 1, 0, 1)
    return output

def get_benchmark_curves_spline(input):

    input_dates = input.index.values
    all_results = []

    for x in input_dates: 
        this_input = input.loc[x]
        rate_1mo = this_input["1mo"]
        rate_3mo = this_input["3mo"]
        rate_6mo = this_input["6mo"]

        # the short slope is the slope in the domain [1 month, 3 months] in the tenor space, the rates for tenors longer than 3 months are also extrapolated using this slope:
        slope2s = (rate_3mo - rate_1mo) / two_month_in_years 
        # the long slope is the slope in the domain [3 month, 6 months] in the tenor space, the rates for tenors shorter than 1 month are also extrapolated using this slope:
        slope2l = (rate_6mo - rate_3mo) / three_months_in_years
        # alpha2 is the intercept: it is solved by means of extrapolation from 1 month tenor to 0 tenor (rate_1mo = intercept + slope * delta(1mo))
        alpha2 = rate_1mo - slope2s * one_month_in_years 

        # times to risk horizons in years: 
        delta_1bd = ((x + BDay(1)) - x).days /360.0
        delta_5bd = ((x + BDay(5)) - x).days / 360.0
        delta_10bd = ((x + BDay(10)) - x).days /360.0

        time_to_risk_hor_1d_add_1mo = delta_1bd + one_month_in_years
        time_to_risk_hor_5d_add_1mo = delta_5bd + one_month_in_years
        time_to_risk_hor_10d_add_1mo = delta_10bd + one_month_in_years

        time_to_risk_hor_1d_add_3mo = delta_1bd + three_months_in_years
        time_to_risk_hor_5d_add_3mo = delta_5bd + three_months_in_years
        time_to_risk_hor_10d_add_3mo = delta_10bd + three_months_in_years

        time_to_risk_hor_1d_add_6mo = delta_1bd + six_months_in_years
        time_to_risk_hor_5d_add_6mo = delta_5bd + six_months_in_years
        time_to_risk_hor_10d_add_6mo = delta_10bd + six_months_in_years

        this_row = ["1mo", x, rate_1mo, alpha2, slope2s, slope2l]

        # The predicted rates on the risk horizon x for the tenor T are computed by: 
        # 1 + time_in_years(x + T) * rate(x + T) = (1 + predicted_rate * time_in_years(T)) * (1 + time_in_years(x) * rate(x))
        # 
        # 1 : In the case T = tenor = 1mo = 1/12, and risk_hor = x = 1bday, the predicted_rate (variable name rate) is solved by
        rate = 12 * ((1.0 + time_to_risk_hor_1d_add_1mo * ( rate_1mo + slope2s * delta_1bd)) / (1.0 + delta_1bd * (alpha2 + slope2s * delta_1bd)) - 1.0)
        this_row.append(rate)

        # 2 : tenor = 1mo, risk_hor = 5day        
        rate = 12 * ((1.0 + time_to_risk_hor_5d_add_1mo * ( rate_1mo +  slope2s * delta_5bd)) / (1.0 + delta_5bd * (alpha2 + slope2s * delta_5bd)) - 1.0)
        this_row.append( rate)

        # 3 : tenor = 1mo, risk_hor = 10day        
        rate = 12 * ((1.0 + time_to_risk_hor_10d_add_1mo * ( rate_1mo  + slope2s * delta_10bd)) / (1.0 + delta_10bd * (alpha2 + slope2s * delta_10bd)) - 1.0)
        this_row.append( rate)
        all_results.append(this_row)

        this_row = ["3mo", x, rate_3mo, alpha2, slope2s, slope2l]

        # 4 : tenor = 3mo, risk_hor = 1day
        rate = 4 * ((1.0 + time_to_risk_hor_1d_add_3mo * ( rate_3mo + slope2l * delta_1bd)) / (1.0 + delta_1bd * (alpha2 + slope2s * delta_1bd)) - 1.0)
        this_row.append( rate)

        # 5 : tenor = 3mo, risk_hor = 5day        
        rate = 4 * ((1.0 + time_to_risk_hor_5d_add_3mo * ( rate_3mo +  slope2l * delta_5bd)) / (1.0 + delta_5bd * (alpha2 + slope2s * delta_5bd)) - 1.0)
        this_row.append( rate)

        # 6 : tenor = 3mo, risk_hor = 10day        
        rate = 4 * ((1.0 + time_to_risk_hor_10d_add_3mo * (rate_3mo + slope2l * delta_10bd)) / (1.0 + delta_10bd * (alpha2 + slope2s * delta_10bd)) - 1.0)
        this_row.append( rate)
        all_results.append(this_row)

        this_row = ["6mo", x,  rate_6mo, alpha2, slope2s, slope2l]

        # 7 : tenor = 6mo, risk_hor = 1day
        rate = 2 * ((1.0 + time_to_risk_hor_1d_add_6mo * ( rate_6mo + slope2l * delta_1bd)) / (1.0 + delta_1bd * (alpha2 + slope2s * delta_1bd)) - 1.0)
        this_row.append(rate)

        # 8 : tenor = 6mo, risk_hor = 5day        
        rate = 2 * ((1.0 + time_to_risk_hor_5d_add_6mo * ( rate_6mo + slope2l * delta_5bd)) / (1.0 + delta_5bd * (alpha2 + slope2s * delta_5bd)) - 1.0)
        this_row.append(rate)

        # 9 : tenor = 6mo, risk_hor = 10day        
        rate = 2 * ((1.0 + time_to_risk_hor_10d_add_6mo * ( rate_6mo + slope2l * delta_10bd)) / (1.0 + delta_10bd * (alpha2 + slope2s * delta_10bd)) - 1.0)
        this_row.append(rate)
        all_results.append(this_row)

    output = pd.DataFrame(all_results, columns = ["tenor", "date", "rate", "alpha", "slopes", "slopel", "rate_roll_1d", "rate_roll_5d", "rate_roll_10d"])
    output["model"] = "benchmark_2"
    return output

def get_benchmark_curves(input): 

    input_dates = input.index.values
    all_results = []
    for x in input_dates: 
        this_input = input.loc[x]
        rate_1mo = this_input["1mo"]
        rate_6mo = this_input["6mo"]
        rate_3mo = this_input["3mo"]

        # regression to fit the dependence: time to matirity vs. rate (also intercept is fitted, therefore the constant 1's)
        mod = sm.OLS([rate_1mo, rate_3mo, rate_6mo], [[1.0, one_month_in_years],[1.0, three_months_in_years],[1.0, six_months_in_years]])
        res = mod.fit()

        alpha = res.params[0]
        slope = res.params[1]
       
        curve_1mo = alpha + slope * one_month_in_years
        curve_3mo = alpha + slope * three_months_in_years
        curve_6mo = alpha + slope * six_months_in_years

        delta_1bd = ((x + BDay(1)) - x).days / 360.0
        delta_5bd = ((x + BDay(5)) - x).days / 360.0
        delta_10bd = ((x + BDay(10)) - x).days / 360.0

        time_to_risk_hor_1d_add_1mo = delta_1bd + one_month_in_years
        time_to_risk_hor_5d_add_1mo = delta_5bd + one_month_in_years
        time_to_risk_hor_10d_add_1mo = delta_10bd + one_month_in_years

        time_to_risk_hor_1d_add_3mo = delta_1bd + three_months_in_years
        time_to_risk_hor_5d_add_3mo = delta_5bd + three_months_in_years
        time_to_risk_hor_10d_add_3mo = delta_10bd + three_months_in_years

        time_to_risk_hor_1d_add_6mo = delta_1bd + six_months_in_years
        time_to_risk_hor_5d_add_6mo = delta_5bd + six_months_in_years
        time_to_risk_hor_10d_add_6mo = delta_10bd + six_months_in_years

        this_row = ["1mo", x, curve_1mo, alpha, slope]

        # The predicted rates on the risk horizon x for the tenor T are computed by: 
        # 1 + time_in_years(x + T) * rate(x + T) = (1 + predicted_rate * time_in_years(T)) * (1 + time_in_years(x) * rate(x))
        # 
        # 1 : In the case T = tenor = 1mo = 1/12, and risk_hor = x = 1bday, the predicted_rate (variable name rate) is solved by
        rate = 12 * ((1.0 + time_to_risk_hor_1d_add_1mo * ( curve_1mo + slope * delta_1bd)) / (1.0 + delta_1bd * (alpha + slope * delta_1bd)) - 1.0) 
        this_row.append(rate)
        
        # 2 : tenor = 1mo, risk_hor = 5day        
        rate = 12 * ((1.0 + time_to_risk_hor_5d_add_1mo * ( curve_1mo +  slope * delta_5bd)) / (1.0 + delta_5bd * (alpha + slope * delta_5bd)) - 1.0)
        this_row.append( rate)

        # 3 : tenor = 1mo, risk_hor = 10day        
        rate = 12 * ((1.0 + time_to_risk_hor_10d_add_1mo * ( curve_1mo  + slope * delta_10bd)) / (1.0 + delta_10bd * (alpha + slope * delta_10bd)) - 1.0)
        this_row.append( rate)
        all_results.append(this_row)
        this_row = ["3mo", x, curve_3mo, alpha, slope]

        # 4 : tenor = 3mo, risk_hor = 1day
        rate = 4 * ((1.0 + time_to_risk_hor_1d_add_3mo * ( curve_3mo + slope * delta_1bd)) / (1.0 + delta_1bd * (alpha + slope * delta_1bd)) - 1.0)
        this_row.append( rate)

        # 5 : tenor = 3mo, risk_hor = 5day        
        rate = 4 * ((1.0 + time_to_risk_hor_5d_add_3mo * (curve_3mo +  slope * delta_5bd)) / (1.0 + delta_5bd * (alpha + slope * delta_5bd)) - 1.0)
        this_row.append( rate)

        # 6 : tenor = 3mo, risk_hor = 10day        
        rate = 4 * ((1.0 + time_to_risk_hor_10d_add_3mo * (curve_3mo + slope * delta_10bd)) / (1.0 + delta_10bd * (alpha + slope * delta_10bd)) - 1.0)
        this_row.append( rate)   
        all_results.append(this_row)
        this_row = ["6mo", x,  curve_6mo, alpha, slope]

        # 7 : tenor = 6mo, risk_hor = 1day
        rate = 2 * ((1.0 + time_to_risk_hor_1d_add_6mo * ( curve_6mo + slope * delta_1bd)) / (1.0 + delta_1bd * (alpha + slope * delta_1bd)) - 1.0)
        this_row.append(rate)

        # 8 : tenor = 6mo, risk_hor = 5day        
        rate = 2 * ((1.0 + time_to_risk_hor_5d_add_6mo * ( curve_6mo + slope * delta_5bd)) / (1.0 + delta_5bd * (alpha + slope * delta_5bd)) - 1.0)
        this_row.append(rate)

        # 9 : tenor = 6mo, risk_hor = 10day        
        rate = 2 * ((1.0 + time_to_risk_hor_10d_add_6mo * ( curve_6mo + slope * delta_10bd)) / (1.0 + delta_10bd * (alpha + slope * delta_10bd)) - 1.0)
        this_row.append(rate)
        all_results.append(this_row)

    output = pd.DataFrame(all_results, columns = ["tenor", "date", "rate", "alpha", "slope", "rate_roll_1d", "rate_roll_5d", "rate_roll_10d"])


    output["model"] = "benchmark_1"
    return output

def align_dates(frame1, frame2, filter = None):
    frame1["date"] = pd.to_datetime(frame1["date"])
    frame2["date"] = pd.to_datetime(frame2["date"])
    all_dates = pd.DataFrame()
    frame_1_index = frame1.index.unique().values    
    frame_2_index = frame2.index.unique().values
    for x in frame_1_index: 
        this_dates = pd.DataFrame(frame1.loc[x]["date"].unique(), columns = ["date"])
        if len(all_dates) == 0: 
            all_dates = this_dates
        else: 
            all_dates = all_dates.merge(this_dates, on = ["date"], how = 'inner')
    for x in frame_2_index: 
        this_dates = pd.DataFrame(frame2.loc[x]["date"].unique(), columns = ["date"])
        all_dates = all_dates.merge(this_dates, on = ["date"], how = 'inner')
    if not filter is None: 
        filter_dates = pd.DataFrame(filter["date"].unique(), columns = ["date"])
        all_dates = all_dates.merge(filter_dates, on = ["date"], how = 'inner')
    frame1 = frame1.loc[frame1["date"].isin(all_dates["date"])]  
    frame2 = frame2.loc[frame2["date"].isin(all_dates["date"])]  
    return frame1, frame2

def test_numerical_accuracy_one_case(deltas, curves, tenor, horizon, window): 

    curves["date"] = pd.to_datetime(curves["date"])

    delta_column = "delta_" + horizon
    rate_roll_column = "rate_roll_" + horizon

    this_deltas = deltas.loc["observed", tenor].sort_values("date")
    this_benchmark_1 = curves.loc["benchmark_1", tenor].sort_values("date")
    this_benchmark_2 = curves.loc["benchmark_2", tenor].sort_values("date")
    this_benchmark_3 = curves.loc["benchmark_3", tenor].sort_values("date")
    this_model = curves.loc["1", tenor].sort_values("date")

    this_deltas = this_deltas[["date", delta_column]].dropna()
    this_benchmark_1["bench1_delta"] = this_benchmark_1[rate_roll_column] - this_benchmark_1["rate"]
    this_benchmark_2["bench2_delta"] = this_benchmark_2[rate_roll_column] - this_benchmark_2["rate"]
    this_benchmark_3["bench3_delta"] = this_benchmark_3[rate_roll_column] - this_benchmark_3["rate"]
    this_model["model_delta"] = this_model[rate_roll_column] - this_model["rate"]

    test_frame_1 = this_benchmark_1[["date","bench1_delta"]]
    test_frame_2 = this_benchmark_2[["date", "bench2_delta"]]
    test_frame_3 = this_benchmark_3[["date", "bench3_delta"]]
    this_model = this_model[["date", "model_delta"]]

    this_test = this_deltas.merge(test_frame_1, on = 'date').merge(test_frame_2, on = 'date').merge(test_frame_3, on = 'date').merge(this_model, on = 'date')
    this_test["year"] = this_test["date"].apply(lambda x : x.year)
    this_test_years = this_test["year"].unique()
    this_test_years.sort()
    this_test = this_test.sort_values("date")

    this_test["bench1_error"] = np.abs((this_test[delta_column] - this_test["bench1_delta"])) 
    this_test["bench2_error"] = np.abs((this_test[delta_column] - this_test["bench2_delta"]))
    this_test["bench3_error"] = np.abs((this_test[delta_column] - this_test["bench3_delta"]))
    this_test["model_error"] = np.abs((this_test[delta_column] - this_test["model_delta"]))

    # Game theory tests. 
    # 1: The model is more accurate more frequently (p-test included)

    b1col = "bench1_error"
    b2col = "bench2_error"
    b3col = "bench3_error"
    mcol = "model_error"
    
    b1_vs_m_b1 = 0
    b1_vs_m_m = 0
    b2_vs_m_b2 = 0
    b2_vs_m_m = 0
    b3_vs_m_b3 = 0
    b3_vs_m_m = 0

    for x in  this_test_years[0:-1]: 
        this_test_one_year = this_test.loc[this_test["year"] == x]
        # this_test_one_year["b1_vs_m_b1"] = 1 if benchmark 1 has more accurate prediction than model, and 0 otherwise 
        this_test_one_year["b1_vs_m_b1"] = np.where(this_test_one_year[b1col] < this_test_one_year[mcol] , 1, 0)
        # this_test_one_year["b1_vs_m_m"] = 1 if model has more accurate prediction than benchmark 1, and 0 otherwise 
        this_test_one_year["b1_vs_m_m"] = np.where(this_test_one_year[mcol] < this_test_one_year[b1col], 1, 0)
        this_test_one_year["b2_vs_m_b2"] = np.where(this_test_one_year[b2col] < this_test_one_year[mcol], 1, 0)
        this_test_one_year["b2_vs_m_m"] = np.where(this_test_one_year[mcol] < this_test_one_year[b2col], 1, 0)
        this_test_one_year["b3_vs_m_b3"] = np.where(this_test_one_year[b3col] < this_test_one_year[mcol], 1, 0)
        this_test_one_year["b3_vs_m_m"] = np.where(this_test_one_year[mcol] < this_test_one_year[b3col], 1, 0)

        data_size = len(this_test_one_year)
        b1_vs_m_b1_success = this_test_one_year["b1_vs_m_b1"].sum()
        b1_vs_m_m_success = this_test_one_year["b1_vs_m_m"].sum()
        b2_vs_m_b2_success = this_test_one_year["b2_vs_m_b2"].sum()
        b2_vs_m_m_success = this_test_one_year["b2_vs_m_m"].sum()
        b3_vs_m_b3_success = this_test_one_year["b3_vs_m_b3"].sum()
        b3_vs_m_m_success = this_test_one_year["b3_vs_m_m"].sum()
        pvalue1 = binomtest(b1_vs_m_b1_success, n=data_size, p=0.5, alternative='greater').pvalue
        pvalue2 = binomtest(b1_vs_m_m_success, n=data_size, p=0.5, alternative='greater').pvalue
        pvalue3 = binomtest(b2_vs_m_b2_success, n=data_size, p=0.5, alternative='greater').pvalue
        pvalue4 = binomtest(b2_vs_m_m_success, n=data_size, p=0.5, alternative='greater').pvalue
        pvalue5 = binomtest(b3_vs_m_b3_success, n=data_size, p=0.5, alternative='greater').pvalue
        pvalue6 = binomtest(b3_vs_m_m_success, n=data_size, p=0.5, alternative='greater').pvalue
        if pvalue1 < 0.05: 
            b1_vs_m_b1 = b1_vs_m_b1 + 1
        if pvalue2 < 0.05:  
            b1_vs_m_m = b1_vs_m_m + 1
        if pvalue3 < 0.05:  
            b2_vs_m_b2 = b2_vs_m_b2 + 1
        if pvalue4 < 0.05:  
            b2_vs_m_m = b2_vs_m_m + 1
        if pvalue5 < 0.05:  
            b3_vs_m_b3 = b3_vs_m_b3 + 1
        if pvalue6 < 0.05:  
            b3_vs_m_m = b3_vs_m_m + 1

    freq_competition = [tenor, horizon, b1_vs_m_b1, b1_vs_m_m, b2_vs_m_b2, b2_vs_m_m, b3_vs_m_b3, b3_vs_m_m]

    # 2: The model is more accurate more frequently (no statistical significance required)

    b1_vs_m_b1 = 0
    b1_vs_m_m = 0
    b2_vs_m_b2 = 0
    b2_vs_m_m = 0
    b3_vs_m_b3 = 0
    b3_vs_m_m = 0

    for x in  this_test_years[0:-1]: 
        this_test_one_year = this_test.loc[this_test["year"] == x]
        this_test_one_year["b1_vs_m_b1"] = np.where(this_test_one_year[b1col] < this_test_one_year[mcol] , 1, 0)
        this_test_one_year["b1_vs_m_m"] = np.where(this_test_one_year[mcol] < this_test_one_year[b1col], 1, 0)
        this_test_one_year["b2_vs_m_b2"] = np.where(this_test_one_year[b2col] < this_test_one_year[mcol], 1, 0)
        this_test_one_year["b2_vs_m_m"] = np.where(this_test_one_year[mcol] < this_test_one_year[b2col], 1, 0)
        this_test_one_year["b3_vs_m_b3"] = np.where(this_test_one_year[b3col] < this_test_one_year[mcol], 1, 0)
        this_test_one_year["b3_vs_m_m"] = np.where(this_test_one_year[mcol] < this_test_one_year[b3col], 1, 0)

        data_size = len(this_test_one_year)
        b1_vs_m_b1_success = this_test_one_year["b1_vs_m_b1"].sum()
        b1_vs_m_m_success = this_test_one_year["b1_vs_m_m"].sum()
        b2_vs_m_b2_success = this_test_one_year["b2_vs_m_b2"].sum()
        b2_vs_m_m_success = this_test_one_year["b2_vs_m_m"].sum()
        b3_vs_m_b3_success = this_test_one_year["b3_vs_m_b3"].sum()
        b3_vs_m_m_success = this_test_one_year["b3_vs_m_m"].sum()
        if b1_vs_m_b1_success > b1_vs_m_m_success: 
            b1_vs_m_b1 = b1_vs_m_b1 + 1
        if b1_vs_m_b1_success < b1_vs_m_m_success:  
            b1_vs_m_m = b1_vs_m_m + 1
        if b2_vs_m_b2_success > b2_vs_m_m_success:  
            b2_vs_m_b2 = b2_vs_m_b2 + 1
        if b2_vs_m_b2_success < b2_vs_m_m_success:  
            b2_vs_m_m = b2_vs_m_m + 1
        if b3_vs_m_b3_success > b3_vs_m_m_success:  
            b3_vs_m_b3 = b3_vs_m_b3 + 1
        if b3_vs_m_b3_success < b3_vs_m_m_success:  
            b3_vs_m_m = b3_vs_m_m + 1
    freq_competition_no_stats = [tenor, horizon, b1_vs_m_b1, b1_vs_m_m, b2_vs_m_b2, b2_vs_m_m, b3_vs_m_b3, b3_vs_m_m]

    # 3: The model is more accurate in average

    b1_vs_m_b1 = 0
    b1_vs_m_m = 0
    b2_vs_m_b2 = 0
    b2_vs_m_m = 0
    b3_vs_m_b3 = 0
    b3_vs_m_m = 0

    for x in this_test_years[0:-1]:
        this_test_one_year = this_test.loc[this_test["year"] == x]
        b1sample = this_test_one_year[b1col].values
        b2sample = this_test_one_year[b2col].values
        b3sample = this_test_one_year[b3col].values
        msample = this_test_one_year[mcol].values

        if b1sample.mean() < msample.mean(): 
            b1_vs_m_b1 = b1_vs_m_b1 + 1
        if msample.mean() < b1sample.mean(): 
            b1_vs_m_m = b1_vs_m_m + 1
        
        if b2sample.mean() < msample.mean(): 
            b2_vs_m_b2 = b2_vs_m_b2 + 1
        if msample.mean() < b2sample.mean(): 
            b2_vs_m_m = b2_vs_m_m + 1
            
        if b3sample.mean() < msample.mean(): 
            b3_vs_m_b3 = b3_vs_m_b3 + 1
        if msample.mean() < b3sample.mean(): 
            b3_vs_m_m = b3_vs_m_m + 1

    mean_competition = [tenor, horizon, b1_vs_m_b1, b1_vs_m_m, b2_vs_m_b2, b2_vs_m_m, b3_vs_m_b3, b3_vs_m_m]
    
    # Predicted delta standard deviations, in percentage: 

    std_observed = 100.0 * this_test[delta_column].std()
    std_bench1 = 100.0 * this_test["bench1_delta"].std()
    std_bench2 = 100.0 * this_test["bench2_delta"].std()
    std_bench3 = 100.0 * this_test["bench3_delta"].std()
    std_model = 100.0 * this_test["model_delta"].std()
    std_row = [tenor, horizon, std_observed, std_bench1, std_bench2, std_bench3, std_model]

    # Normalized average deltas: 

    normalization_factor = np.abs(this_test[delta_column]).mean()
    average_delta_error1 = (this_test["bench1_error"]).mean() / normalization_factor
    average_delta_error2 = (this_test["bench2_error"]).mean() / normalization_factor
    average_delta_error3 = (this_test["bench3_error"]).mean() / normalization_factor
    average_delta_error_model = (this_test["model_error"]).mean() / normalization_factor

    average_delta_errror = [tenor, horizon, average_delta_error1, average_delta_error2, average_delta_error3, average_delta_error_model]
    
    return std_row, average_delta_errror, freq_competition, freq_competition_no_stats, mean_competition

def test_numerical_accuracy(deltas, all_curves, horizons):
    
    horizons = [str(x) + "d" for x in horizons]
    tenors = ["1mo", "3mo", "6mo"]
    windows = [int(250 * 2 / 3), int(50 * 2 / 3),  int(25 * 2 / 3) ]
    all_stdevs_table = []
    average_delta_errror_table = []
    freq_competition_table = []
    freq_competition_no_stats_table = []
    mean_competition_table = []
    for T in tenors: 
        for t, w in zip(horizons, windows): 
            std_row, average_delta_errror, freq_competition, freq_competition_no_stats, mean_competition = test_numerical_accuracy_one_case(deltas, all_curves, T, t, w)
            all_stdevs_table.append(std_row)
            average_delta_errror_table.append(average_delta_errror)
            freq_competition_table.append(freq_competition)
            freq_competition_no_stats_table.append(freq_competition_no_stats)
            mean_competition_table.append(mean_competition)

    all_stdevs_table = pd.DataFrame(all_stdevs_table, columns = ["Tenor", "Horizon", "Observed", "Benchmark1", "Benchmark2", "Benchmark3", "Model"])
    average_delta_errror_table = pd.DataFrame(average_delta_errror_table, columns = ["Tenor", "Horizon", "Benchmark1", "Benchmark2", "Benchmark3", "Model"])
    freq_competition_table = pd.DataFrame(freq_competition_table, columns = ["Tenor", "Horizon", "B1_M_B1", "B1_M_M", "B2_M_B2", "B2_M_M", "B3_M_B3", "B3_M_M"])
    freq_competition_no_stats_table = pd.DataFrame(freq_competition_no_stats_table, columns = ["Tenor", "Horizon", "B1_M_B1", "B1_M_M", "B2_M_B2", "B2_M_M", "B3_M_B3", "B3_M_M"])
    mean_competition_table = pd.DataFrame(mean_competition_table, columns = ["Tenor", "Horizon", "B1_M_B1", "B1_M_M", "B2_M_B2", "B2_M_M", "B3_M_B3", "B3_M_M"])

    return all_stdevs_table, average_delta_errror_table, freq_competition_table, freq_competition_no_stats_table, mean_competition_table


