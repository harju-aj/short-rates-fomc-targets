
import pandas as pd
import os
import matplotlib
import numpy as np
from pandas.tseries.offsets import BDay
import matplotlib.pyplot as plt

from detstrc_py import data_helpers
from detstrc_py import calibration
from detstrc_py import expected_rates
from detstrc_py import expected_rates_study
from detstrc_py import rate_change_study
from detstrc_py import dependency_study

from detstrc_py.helpers import get_bdays

matplotlib.use('TkAgg')

import warnings
warnings.filterwarnings('ignore')

first_date = "2013-01-01"

def table_to_latex(table, numeric_coumns_after): 
    nrows = len(table)
    column_names = table.columns
    for i in np.arange(nrows):
        this_row_string = ""
        this_row = table.iloc[i]
        for xx in column_names[0:numeric_coumns_after]:
            this_row_string = this_row_string + str(this_row[xx]) + " & "
        for xx in column_names[numeric_coumns_after:]:
            this_row_string = this_row_string + str(round(this_row[xx] * 100.0)/100.0) + " & "
        this_row_string = this_row_string[0:-2]
        this_row_string = this_row_string + "\\\\"
        print(this_row_string)
    return

def print_column_for_latex(input, column): 
    input_values = input[column].values
    input_indexes = np.arange(1, input_values.shape[0] + 1)
    string_to_print = ""
    for x, y in zip(input_indexes, input_values):
        this_string = str((x,y))
        string_to_print += this_string
   
    print(string_to_print)
    return

def read_data_from_files_and_prepare_inputs(): 

    current_directory = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(current_directory, "data")
    futures_directory = os.path.join(data_directory, "fed-futures-price")
    
    rate_1mo = pd.read_csv(os.path.join(data_directory, "DGS1MO.csv"))[["date", "Close"]].rename({"Close" : "1mo"}, axis = 1)
    rate_3mo = pd.read_csv(os.path.join(data_directory, "DGS3MO.csv"))[["date", "Close"]].rename({"Close" : "3mo"}, axis = 1)
    rate_6mo = pd.read_csv(os.path.join(data_directory, "DGS6MO.csv"))[["date", "Close"]].rename({"Close" : "6mo"}, axis = 1)
    # The input frames mark nans as ".", lets drop and fill later: 
    rate_1mo = rate_1mo.loc[rate_1mo["1mo"] != "."]
    rate_3mo = rate_3mo.loc[rate_3mo["3mo"] != "."]
    rate_6mo = rate_6mo.loc[rate_6mo["6mo"] != "."]
    rate_1mo["1mo"] = pd.to_numeric(rate_1mo["1mo"]) / 100.0
    rate_3mo["3mo"] = pd.to_numeric(rate_3mo["3mo"]) / 100.0
    rate_6mo["6mo"] = pd.to_numeric(rate_6mo["6mo"]) / 100.0
    rates = rate_1mo.merge(rate_3mo, on = ["date"], how = "outer")
    rates = rates.merge(rate_6mo, on = ["date"], how = "outer")

    # fomc dates (announced) 
    # alternative fomc dates (actual)
    fomc_dates = pd.read_csv(os.path.join(data_directory, "fomc-dates.csv"))
    fomc_dates_alt = pd.read_csv(os.path.join(data_directory, "fomc-dates-alternative.csv"))
    fomc_dates["date"] = pd.to_datetime(fomc_dates["fomc-date"], format="%Y%m%d")
    fomc_dates_alt["date"] = pd.to_datetime(fomc_dates_alt["fomc-date"], format="%Y%m%d")

    # effr rates -- the inputs are in pct units, and transformed to decimal units: 
    # effr_rates = pd.read_csv(os.path.join(data_directory, "effr.csv"))
    effr_rates = pd.read_csv(os.path.join(data_directory, "DFF.csv"))
    effr_rates = data_helpers.smooth_effrs(effr_rates, fomc_dates)
    effr_rates["date"] = pd.to_datetime(effr_rates["date"])
    effr_rates["effr"] = effr_rates["effr"] / 100.0

    # futures prices: 
    futures_prices = data_helpers.process_future_prices(futures_directory)
    futures_prices = futures_prices.loc[futures_prices["date"] >= first_date]

    # missing rates and prices are backfilled: 
    dates = data_helpers.get_dates(pd.to_datetime(first_date) - BDay(250), futures_prices["date"].max(), "BDay")
    calendar_dates = data_helpers.get_dates(pd.to_datetime(first_date) - BDay(250), futures_prices["date"].max(), "CDay")

    futures_prices = data_helpers.forward_fill_columns(futures_prices, dates, ["price", "year", "month"], "contract" )
    futures_prices = futures_prices.drop("contract", axis = 1) # contract column is not needed anymore.

    # to price Fed funds futures, we need EFFR to be forward filled over weekends and holidays.
    # EFFR is available on eacn calendar date (not only business dates).
    effr_rates = data_helpers.forward_fill_columns(effr_rates, calendar_dates, ["effr"])
    rates = data_helpers.forward_fill_columns(rates, dates, ["1mo", "3mo", "6mo"])
    return rates, effr_rates, futures_prices, fomc_dates, fomc_dates_alt


def get_predicted_model_rates(calibration_results): 
    
    # effrs are the market quotes. 
    effrs = calibration_results[["date", "effr"]].drop_duplicates(subset=['date'], keep = 'first')
    # overnight_rates are the overnight rates calibrated for the benchmark 3. 
    overnight_rates = calibration_results[["date", "overnight_rate"]].drop_duplicates(subset=['date'], keep = 'first')
    # parameters frame has the effr rates, and the jumps for the calibrated model. 
    parameters = effrs.merge(calibration_results.pivot(index = 'date', columns = 'fomc')["jump"].reset_index(), on = 'date')
    # parameters_2 frame has the calibrated overnight rates, and the jumps for the benchmark 3.
    parameters_2 = overnight_rates.merge(calibration_results.pivot(index = 'date', columns = 'fomc')["jump_bench"].reset_index(), on = 'date')
    # fomcs have the dates for the 5 upcoming meetings.
    fomcs = calibration_results.pivot(index = ['date'], columns = 'fomc')["fomc_date"].reset_index()

    # Compute the model predicted rates: 
    predicted_rates_1 = expected_rates.predict_rates(parameters, fomcs)
    predicted_rates_2 = expected_rates.predict_rates(parameters_2, fomcs)
    
    predicted_rates_1 = predicted_rates_1.reset_index()
    predicted_rates_1["model"] = "1"
    predicted_rates_1 = predicted_rates_1.set_index(["model", "tenor"])

    predicted_rates_2 = predicted_rates_2.reset_index()
    predicted_rates_2["model"] = "benchmark_3"
    predicted_rates_2 = predicted_rates_2.set_index(["model", "tenor"])

    # Compute the model predicted rates (linear estimates): 
    predicted_rates_1_linear = expected_rates.predict_rates(parameters, fomcs, True)
    predicted_rates_2_linear = expected_rates.predict_rates(parameters_2, fomcs, True)
    
    predicted_rates_1_linear = predicted_rates_1_linear.reset_index()
    predicted_rates_1_linear["model"] = "1"
    predicted_rates_1_linear = predicted_rates_1_linear.set_index(["model", "tenor"])

    predicted_rates_2_linear = predicted_rates_2_linear.reset_index()
    predicted_rates_2_linear["model"] = "benchmark_3"
    predicted_rates_2_linear = predicted_rates_2_linear.set_index(["model", "tenor"])
    
    all_predicted_rates = pd.concat([predicted_rates_1, predicted_rates_2])
    all_predicted_rates_linear = pd.concat([predicted_rates_1_linear, predicted_rates_2_linear])

    return all_predicted_rates, all_predicted_rates_linear

def get_rate_deltas(rates, calibration_results):

    horizons = [1, 5, 10]
    first_parameter_date = calibration_results["date"].min()
    last_parameter_date = calibration_results["date"].max()
    bdays_1d_calendar = get_bdays(first_parameter_date, last_parameter_date, 1)
    bdays_5d_calendar = get_bdays(first_parameter_date, last_parameter_date, 5)
    bdays_10d_calendar = get_bdays(first_parameter_date, last_parameter_date, 10)
    deltas_1d, deltas_5d, deltas_10d = expected_rates_study.get_rate_deltas(rates, [bdays_1d_calendar, bdays_5d_calendar, bdays_10d_calendar], horizons)
    delta_observed = deltas_1d.merge(deltas_5d, on = ["date", "tenor"], how = "outer")
    delta_observed = delta_observed.merge(deltas_10d, on = ["date", "tenor"], how = "outer")
    delta_observed = delta_observed.sort_values(["tenor", "date"])
    delta_observed["model"] = "observed"
    delta_observed = delta_observed.set_index(["model", "tenor"])
    return delta_observed

def get_benchmark_rates(rates):

    bench_curves = expected_rates_study.get_benchmark_curves(rates) # dec
    bench_curves2 = expected_rates_study.get_benchmark_curves_spline(rates) # dec
    bench_curves = pd.concat([bench_curves, bench_curves2])
    bench_curves["date"] = pd.to_datetime(bench_curves["date"])
    bench_curves = bench_curves.set_index(["model", "tenor"])
    return bench_curves

def do_expected_rates_analysis(rates, delta_observed, predicted_rates, fomc_dates): 

    risk_horizons = [1,5,10]
    rates_input = rates.set_index("date")
    
    # The fomc indicators carry information about the time left to a FOMC meeting. This is used for the filtering later. 
    fomc_indicators = expected_rates_study.get_fomc_indicators(delta_observed, fomc_dates)

    bench_rates = get_benchmark_rates(rates_input)
    bench_rates, predicted_rates = expected_rates_study.align_dates(bench_rates, predicted_rates)
    all_predicted_rates = pd.concat([bench_rates, predicted_rates])
    all_stdevs_table, average_delta_error_table, freq_competition_table, freq_competition_no_stats_table, mean_competition_table = expected_rates_study.test_numerical_accuracy(delta_observed, all_predicted_rates, risk_horizons)
        
    # The case: FOMC meeting 10 bdays or more ahead
    filter_frame_fomc_far = fomc_indicators.loc[fomc_indicators["fomc_10d_window"] == 0]
    bench_rates_fomc_far, predicted_rates_fomc_far = expected_rates_study.align_dates(bench_rates, predicted_rates, filter_frame_fomc_far)
    all_predicted_rates_fomc_far = pd.concat([bench_rates_fomc_far, predicted_rates_fomc_far])
    all_stdevs_table_fomc_far, average_delta_error_table_fomc_far, freq_competition_table_fomc_far, freq_competition_no_stats_table_fomc_far, mean_competition_table_fomc_far = expected_rates_study.test_numerical_accuracy(delta_observed, all_predicted_rates_fomc_far, risk_horizons)
    
    # The case: FOMC meeting earlier than 10 bdays ahead
    filter_frame_fomc_close = fomc_indicators.loc[fomc_indicators["fomc_10d_window"] == 1]
    bench_rates_fomc_close, predicted_rates_fomc_close = expected_rates_study.align_dates(bench_rates, predicted_rates, filter_frame_fomc_close)
    all_predicted_rates_fomc_close = pd.concat([bench_rates_fomc_close, predicted_rates_fomc_close])
    all_stdevs_table_fomc_close, average_delta_error_table_fomc_close, freq_competition_table_fomc_close, freq_competition_no_stats_table_fomc_close, mean_competition_table_fomc_close = expected_rates_study.test_numerical_accuracy(delta_observed, all_predicted_rates_fomc_close, risk_horizons)
    
    print("Table 1. Standard deviations of the rate change and predicted rate change time series")
    print(all_stdevs_table)
    print("\n")
    print("Table 2. Average delta error table")
    print(average_delta_error_table)
    print("\n")
    print("Table 3. Average delta error table - FOMC meeting 10 bdays or more ahead")
    print(average_delta_error_table_fomc_far)
    print("\n")
    print("Table 4. Average delta error table - FOMC meeting earlier than 10 bdays")
    print(average_delta_error_table_fomc_close)
    print("\n")
    print("Table 5-a. Frequency competition table")
    print(freq_competition_no_stats_table)
    print("\n")
    print("Table 5-b. Frequency competition table -- stats included")
    print(freq_competition_table)
    print("\n")
    return 

def do_dependency_analysis(delta_observed, parameters, shift):
    dependency_study.study_days_to_fomc_x_prediction_error_dependency(parameters, shift)
    delta_stdevs_pre2020 = dependency_study.compute_delta_stdevs(parameters, delta_observed.loc[delta_observed['date'] < '2020-01-01'] )
    delta_stdevs_post2020 = dependency_study.compute_delta_stdevs(parameters, delta_observed.loc[delta_observed['date'] >= '2020-01-01'])
    print("Table 6. Standard deviations in different buckets")
    print(10000 * delta_stdevs_pre2020)   # in bps
    print(10000 * delta_stdevs_post2020)  # in bps
    print("\n")    
    return

def do_rate_change_explanatory_analysis(effr_rates, delta_observed, predicted_rates_linear, parameters):

    horizons = [1, 5, 10]
    tenors = ["1mo", "3mo", "6mo"]
    deltas = delta_observed.copy().reset_index()
    horizon_names = ["delta_" + str(x) + "d" for x in horizons]
    delta_observed_list = [deltas[["tenor", "date", horizon]].dropna() for horizon in horizon_names]

    # First plot Image 3: 5day horizon stdevs for 3mo rate: 
    horizon = 5
    delta_observed_5d = delta_observed_list[1]
    delta_column = "delta_" + str(horizon) + "d"
    jump_factors, nojump_factors, jump_sensitivities, nojump_sensitivities = rate_change_study.get_risk_factors_one_horizon(parameters, delta_observed_5d, effr_rates, horizon, 'alt_fomc_date', 'alt_days_to_fomc')
    jump_residuals, nojump_residuals = rate_change_study.get_factor_residual_one_horizon(delta_observed_5d, predicted_rates_linear, jump_factors, nojump_factors, jump_sensitivities, nojump_sensitivities, horizon)
    all_residuals = rate_change_study.fit_factor_residual_level_changes(jump_residuals, nojump_residuals)

    rolling_window = 100
    for x in tenors: 
        all_residuals[x] = all_residuals[x].sort_values("date")
        all_residuals[x][delta_column + "_std"] = all_residuals[x][delta_column].rolling(rolling_window).std()
        all_residuals[x]["risky_delta_std"] = all_residuals[x]["risky_delta"].rolling(rolling_window).std()
        all_residuals[x]["factor_residual_std"] = all_residuals[x]["factor_residual"].rolling(rolling_window).std()
        all_residuals[x]["residual_std"] = all_residuals[x]["residual"].rolling(rolling_window).std()
        if x == "3mo": 
            all_residuals[x].plot(x = "date", y = [delta_column + "_std", "risky_delta_std", "factor_residual_std", "residual_std"])
            plt.show()
            """
            X = all_residuals[x].copy()
            X = X.dropna()
            X[delta_column + "_std"] *= 10000
            X["risky_delta_std"] *= 10000
            X["factor_residual_std"] *= 10000
            X["residual_std"] *= 10000
            print_column_for_latex(X, delta_column + "_std")
            print_column_for_latex(X, "risky_delta_std")
            print_column_for_latex(X, "factor_residual_std")
            print_column_for_latex(X, "residual_std")
            """
            
    # Then produce the tables: 7, 8, 9, 10: 
    results = []
    results_no_jump = []
    results_jump = []
    kurt_results = []
    kurt_results_pre_covid = []
    kurt_results_post_covid = []
    correlations = []
    all_stdevs_annually = pd.DataFrame()
    for horizon, delta_frame in zip(horizons, delta_observed_list): 
        delta_column = "delta_" + str(horizon) + "d"
        this_jump_factors, this_nojump_factors, this_jump_sensitivities, this_nojump_sensitivities = rate_change_study.get_risk_factors_one_horizon(parameters, delta_frame , effr_rates, horizon, 'alt_fomc_date', 'alt_days_to_fomc')
        jump_residuals, nojump_residuals = rate_change_study.get_factor_residual_one_horizon(delta_frame, predicted_rates_linear, this_jump_factors, this_nojump_factors, this_jump_sensitivities, this_nojump_sensitivities, horizon)
        this_all_residuals = rate_change_study.fit_factor_residual_level_changes(jump_residuals, nojump_residuals)
        for xx in tenors:
            residuals = this_all_residuals[xx]
            residuals_no_jump = this_all_residuals[xx].loc[this_all_residuals[xx]["jump"] == 0]
            residuals_jump = this_all_residuals[xx].loc[this_all_residuals[xx]["jump"] == 1]
            stdevs = [xx, 
                delta_column,
                residuals[delta_column].std() * 10000,
                residuals["risky_delta"].std() * 10000,
                residuals["factor_residual"].std() * 10000,
                residuals["residual"].std() * 10000,
                residuals["factor_delta"].std() * 10000,
                residuals["level_factor"].std() * 10000 ]
            results.append(stdevs)
            stdevs_no_jump = [xx, 
                delta_column,
                residuals_no_jump[delta_column].std() * 10000,
                residuals_no_jump["risky_delta"].std() * 10000,
                residuals_no_jump["factor_residual"].std() * 10000,
                residuals_no_jump["residual"].std() * 10000,
                residuals_no_jump["factor_delta"].std() * 10000,
                residuals_no_jump["level_factor"].std() * 10000 ]
            results_no_jump.append(stdevs_no_jump)
            stdevs_jump = [xx, 
                delta_column,
                residuals_jump[delta_column].std() * 10000,
                residuals_jump["risky_delta"].std() * 10000,
                residuals_jump["factor_residual"].std() * 10000,
                residuals_jump["residual"].std() * 10000,
                residuals_jump["factor_delta"].std() * 10000,
                residuals_jump["level_factor"].std() * 10000 ]
            results_jump.append(stdevs_jump)
            residuals_pre_covid = residuals.loc[residuals['date'] < '2020-01-01']
            residuals_post_covid = residuals.loc[residuals['date'] >= '2020-01-01']
            kurt = [xx, 
                delta_column,
                residuals[delta_column].kurt(),
                residuals["risky_delta"].kurt(),
                residuals["factor_residual"].kurt(),
                residuals["residual"].kurt(),
                residuals["factor_delta"].kurt(),
                residuals["level_factor"].kurt() ]
            kurt_results.append(kurt)
            kurt_pre_covid = [xx, 
                delta_column,
                residuals_pre_covid[delta_column].kurt(),
                residuals_pre_covid["risky_delta"].kurt(),
                residuals_pre_covid["factor_residual"].kurt(),
                residuals_pre_covid["residual"].kurt(),
                residuals_pre_covid["factor_delta"].kurt(),
                residuals_pre_covid["level_factor"].kurt() ]
            kurt_results_pre_covid.append(kurt_pre_covid)
            kurt_post_covid = [xx, 
                delta_column,
                residuals_post_covid[delta_column].kurt(),
                residuals_post_covid["risky_delta"].kurt(),
                residuals_post_covid["factor_residual"].kurt(),
                residuals_post_covid["residual"].kurt(),
                residuals_post_covid["factor_delta"].kurt(),
                residuals_post_covid["level_factor"].kurt() ]
            kurt_results_post_covid.append(kurt_post_covid)
            correlations.append([xx, 
                delta_column,
                residuals[['level_factor', 'factor_delta', 'residual']].corr().loc['factor_delta', 'level_factor'],
                residuals[['level_factor', 'factor_delta', 'residual']].corr().loc['factor_delta', 'residual'],
                residuals[['level_factor', 'factor_delta', 'residual']].corr().loc['level_factor', 'residual']
            ])
            residuals['year'] = residuals['date'].dt.year
            deltas_annually = pd.DataFrame(residuals.groupby('year')[delta_column].std(), columns = [delta_column]) * 10000
            deltas_annually.columns = ['delta']
            factor_residuals_annually = pd.DataFrame(residuals.groupby('year')["factor_residual"].std(), columns = ["factor_residual"]) * 10000
            residuals_annually = pd.DataFrame(residuals.groupby('year')["residual"].std(), columns = ["residual"]) * 10000
            stdevs_annually = deltas_annually.join([factor_residuals_annually, residuals_annually]).reset_index()
            stdevs_annually['tenor'] = xx
            stdevs_annually['horizon'] = horizon
            all_stdevs_annually = pd.concat([all_stdevs_annually, stdevs_annually])

    results = pd.DataFrame(data = results, columns = ['tenor', 'horizon', 'delta_std', 'risky_delta_std', 'factor_residual_std', 'residual_std', 'factor_std', 'slope_std'] )
    results_no_jump = pd.DataFrame(data = results_no_jump, columns = ['tenor', 'horizon', 'delta_std', 'risky_delta_std', 'factor_residual_std', 'residual_std', 'factor_std', 'slope_std'] )
    results_jump = pd.DataFrame(data = results_jump, columns = ['tenor', 'horizon', 'delta_std', 'risky_delta_std', 'factor_residual_std', 'residual_std', 'factor_std', 'slope_std']  )
    correlations = pd.DataFrame(data = correlations, columns = ['tenor', 'horizon', 'factor_delta-level_factor', 'factor_delta-residual', 'level_factor-residual'] )
    kurt_results = pd.DataFrame(data = kurt_results, columns = ['tenor', 'horizon', 'delta_kurt', 'risky_delta_kurt', 'factor_residual_kurt', 'residual_kurt', 'factor_kurt', 'level_kurt'] )
    kurt_results_pre_covid = pd.DataFrame(data = kurt_results_pre_covid, columns = ['tenor', 'horizon', 'delta_kurt', 'risky_delta_kurt', 'factor_residual_kurt', 'residual_kurt', 'factor_kurt', 'level_kurt'] )
    kurt_results_post_covid = pd.DataFrame(data = kurt_results_post_covid, columns = ['tenor', 'horizon', 'delta_kurt', 'risky_delta_kurt', 'factor_residual_kurt', 'residual_kurt', 'factor_kurt', 'level_kurt'] )

    results = results.sort_values(['tenor'])
    results_no_jump = results_no_jump.sort_values(['tenor'])
    results_jump = results_jump.sort_values(['tenor'])
    correlations = correlations.sort_values(['tenor'])
    kurt_results = kurt_results.sort_values(['tenor'])
    kurt_results_pre_covid = kurt_results_pre_covid.sort_values(['tenor'])
    kurt_results_post_covid = kurt_results_post_covid.sort_values(['tenor'])


    all_stdevs_annually_delta = all_stdevs_annually.pivot(index = ['tenor', 'horizon'], columns = 'year')["delta"].reset_index().drop(2024, axis =1 )
    all_stdevs_annually_factor_residual = all_stdevs_annually.pivot(index = ['tenor', 'horizon'], columns = 'year')["factor_residual"].reset_index().drop(2024, axis =1 )
    all_stdevs_annually_residual = all_stdevs_annually.pivot(index = ['tenor', 'horizon'], columns = 'year')["residual"].reset_index().drop(2024, axis =1 )
    all_stdevs_annually_delta.columns = all_stdevs_annually_delta.columns.rename(None)
    all_stdevs_annually_factor_residual.columns = all_stdevs_annually_factor_residual.columns.rename(None)
    all_stdevs_annually_residual.columns = all_stdevs_annually_residual.columns.rename(None)

    print("Table 7: standard deviations")
    print(results)
    print(results_no_jump)
    print(results_jump)
    print("\n")
    print("Table 8: annualized standard deviations")
    print(all_stdevs_annually_delta)
    print(all_stdevs_annually_factor_residual)
    print(all_stdevs_annually_residual)
    print("\n")
    print("Table 9: correlations")
    print(correlations)
    print("\n")
    print("Table 10: excess kurtosis : pre covid")
    print(kurt_results_pre_covid)
    print("\n")
    print("Table 10: excess kurtosis : post covid")
    print(kurt_results_post_covid)
    print("\n")

    variance_5d_1mo_delta = (results.iloc[1,2] / 10000) ** 2
    variance_10d_1mo_delta = (results.iloc[2,2] / 10000) ** 2
    variance_5d_3mo_delta = (results.iloc[4,2] / 10000) ** 2
    variance_10d_3mo_delta = (results.iloc[5,2] / 10000) ** 2
    variance_5d_6mo_delta = (results.iloc[7,2] / 10000) ** 2
    variance_10d_6mo_delta = (results.iloc[8,2] / 10000) ** 2

    variance_5d_1mo_model = (results.iloc[1,4] / 10000) ** 2
    variance_10d_1mo_model = (results.iloc[2,4] / 10000) ** 2
    variance_5d_3mo_model = (results.iloc[4,4] / 10000) ** 2
    variance_10d_3mo_model = (results.iloc[5,4] / 10000) ** 2
    variance_5d_6mo_model = (results.iloc[7,4] / 10000) ** 2
    variance_10d_6mo_model = (results.iloc[8,4] / 10000) ** 2

    variance_reduction_5d = [(variance_5d_1mo_delta - variance_5d_1mo_model) / variance_5d_1mo_delta, 
                             (variance_5d_3mo_delta - variance_5d_3mo_model) / variance_5d_3mo_delta, 
                             (variance_5d_6mo_delta - variance_5d_6mo_model) / variance_5d_6mo_delta]
    variance_reduction_10d = [(variance_10d_1mo_delta - variance_10d_1mo_model) / variance_10d_1mo_delta, 
                              (variance_10d_3mo_delta - variance_10d_3mo_model) / variance_10d_3mo_delta, 
                              (variance_10d_6mo_delta - variance_10d_6mo_model) / variance_10d_6mo_delta]
    variance_reduction = pd.DataFrame([variance_reduction_5d, variance_reduction_10d], columns = ['1mo', '3mo', '6mo'], index = ['5d', '10d'])

    print("Variance reductions mentioned in the text:")
    print(variance_reduction * 100) # in pct
    print("\n")
    return

def do_risk_case_studies(effr_rates, delta_observed, predicted_rates_linear, parameters):

    horizon = 5
    horizon_name = "delta_" + str(horizon) + "d" 
    delta_frame = delta_observed.copy().reset_index()[["tenor", "date", horizon_name]].dropna()
    
    # First produce the factors, sensitivities and residuals:
    jump_factors, nojump_factors, jump_sensitivities, nojump_sensitivities = rate_change_study.get_risk_factors_one_horizon(parameters, delta_frame , effr_rates, horizon, 'alt_fomc_date', 'days_to_fomc')
    jump_residuals, nojump_residuals = rate_change_study.get_factor_residual_one_horizon(delta_frame, predicted_rates_linear, jump_factors, nojump_factors, jump_sensitivities, nojump_sensitivities, horizon)
    all_residuals = rate_change_study.fit_factor_residual_level_changes(jump_residuals, nojump_residuals)

    # The level factor sensitivities are also needed for the risk application: 
    for frame in jump_sensitivities.values():
        frame["level"] = 1.0
    for frame in nojump_sensitivities.values():
        frame["level"] = 1.0   
    
    # Index '1' is the model (instaed of the benchmark used earlier)
    predicted_rates_linear_model = predicted_rates_linear.loc['1'].reset_index()
    # phi_5d column has the expected rate changes over the 5d risk window: 
    predicted_rates_linear_model['phi_5d'] = predicted_rates_linear_model['rate_roll_5d'] - predicted_rates_linear_model['rate']

    # pre-covid case:
    test_date = "2019-11-27"
    factor_breakdown_precovid = rate_change_study.get_factor_risk_breakdown(jump_factors, nojump_factors, jump_sensitivities, nojump_sensitivities, all_residuals, test_date, 750)
    factor_breakdown_precovid = predicted_rates_linear_model[['date', 'tenor', 'phi_5d']].merge(factor_breakdown_precovid, on = ['date', 'tenor'])
    factor_breakdown_precovid.iloc[:,2:] = factor_breakdown_precovid.iloc[:,2:] * 10000
    factor_breakdown_precovid.rename( columns  = {'factor 5' : 'trf' }, inplace = True)
    factor_breakdown_precovid['trf'] = factor_breakdown_precovid.iloc[:, 3:8].sum(axis = 1)

    test_date = "2020-02-26"
    factor_breakdown_precovid2 = rate_change_study.get_factor_risk_breakdown(jump_factors, nojump_factors, jump_sensitivities, nojump_sensitivities, all_residuals, test_date, 750)
    factor_breakdown_precovid2 = predicted_rates_linear_model[['date', 'tenor', 'phi_5d']].merge(factor_breakdown_precovid2, on = ['date', 'tenor'])
    factor_breakdown_precovid2.iloc[:,2:] = factor_breakdown_precovid2.iloc[:,2:] * 10000
    factor_breakdown_precovid2.rename( columns  = {'factor 5' : 'trf' }, inplace = True)
    factor_breakdown_precovid2['trf'] = factor_breakdown_precovid2.iloc[:, 3:8].sum(axis = 1)

    jumps_precovid = parameters.loc[parameters['date'] == test_date][['days_to_fomc', 'jump']]
    jumps_precovid['jump'] = jumps_precovid['jump'] * 10000

    # post-covid case 1:
    test_date = "2022-03-16"
    factor_breakdown_postcovid1 = rate_change_study.get_factor_risk_breakdown(jump_factors, nojump_factors, jump_sensitivities, nojump_sensitivities, all_residuals, test_date, 750)
    factor_breakdown_postcovid1 = predicted_rates_linear_model[['date', 'tenor', 'phi_5d']].merge(factor_breakdown_postcovid1, on = ['date', 'tenor'])
    factor_breakdown_postcovid1.iloc[:,2:] = factor_breakdown_postcovid1.iloc[:,2:] * 10000
    factor_breakdown_postcovid1.rename( columns  = {'factor 5' : 'trf' }, inplace = True)
    factor_breakdown_postcovid1['trf'] = factor_breakdown_postcovid1.iloc[:, 3:8].sum(axis = 1)
    jumps_postcovid1 = parameters.loc[parameters['date'] == test_date][['days_to_fomc', 'jump']]
    jumps_postcovid1['jump'] = jumps_postcovid1['jump'] * 10000

    test_date = "2022-09-21"
    factor_breakdown_postcovid2 = rate_change_study.get_factor_risk_breakdown(jump_factors, nojump_factors, jump_sensitivities, nojump_sensitivities, all_residuals, test_date, 750)
    factor_breakdown_postcovid2 = predicted_rates_linear_model[['date', 'tenor', 'phi_5d']].merge(factor_breakdown_postcovid2, on = ['date', 'tenor'])
    factor_breakdown_postcovid2.iloc[:,2:] = factor_breakdown_postcovid2.iloc[:,2:] * 10000
    factor_breakdown_postcovid2.rename( columns  = {'factor 5' : 'trf' }, inplace = True)
    factor_breakdown_postcovid2['trf'] = factor_breakdown_postcovid2.iloc[:, 3:8].sum(axis = 1)

    jumps_postcovid2 = parameters.loc[parameters['date'] == test_date][['days_to_fomc', 'jump']]
    jumps_postcovid2['jump'] = jumps_postcovid2['jump'] * 10000

    pre_covid_risk = pd.concat([factor_breakdown_precovid, factor_breakdown_precovid2,], axis = 0)
    post_covid_risk = pd.concat([factor_breakdown_postcovid1, factor_breakdown_postcovid2], axis = 0)

    print("Table 11:")
    print(pre_covid_risk)
    print("\n")
    print("Table 12:")
    print(post_covid_risk)
    print("\n")
    return

if __name__ == "__main__":

    # The following 3 variables may be given True of False values.
    # This is a variable that can get values True and False (calibration takes long time, but only needs to be done once).
    CALIBRATION_MODE = True
    # The training horizon for the model parameters (7 months) may be adjusted by changing the following:
    # increse (+1 month) or decrase (-1 month).
    BUMP_UP_TRAINING_HORIZON = False
    BUMP_DOWN_TRAINING_HORIZON = False
    # This package does not have other variables that should be manipulated.

    # fomc_dates has the fomc meeting dates that were announced. 
    # fomc_dates_alt has the actual fomc meeting dates. 
    rates, effr_rates, futures_prices, fomc_dates, fomc_dates_alt = read_data_from_files_and_prepare_inputs()
    
    current_directory = os.path.dirname(os.path.abspath(__file__))
    calibration_results_file = os.path.join(current_directory, "calibration_results.csv")
    predicted_rates_file = os.path.join(current_directory, "predicted_rates.csv")
    predicted_linear_rates_file = os.path.join(current_directory, "predicted_linear_rates.csv")

    if CALIBRATION_MODE:     
        calibration_results = calibration.calibrate_parameters(effr_rates, fomc_dates, fomc_dates_alt, futures_prices, rates, BUMP_UP_TRAINING_HORIZON - BUMP_DOWN_TRAINING_HORIZON)
        calibration_results.to_csv(calibration_results_file, index = False)
        # predicted_rates has the rates predicted by the model, and benchmark 3, on the valuation dates, and at the 1-bday, 5-bday and 10-bday risk horizons.
        predicted_rates, predicted_rates_linear = get_predicted_model_rates(calibration_results)
        predicted_rates.to_csv(predicted_rates_file)
        predicted_rates_linear.to_csv(predicted_linear_rates_file)
    else: 
        calibration_results = pd.read_csv(calibration_results_file)
        calibration_results["date"] = pd.to_datetime(calibration_results["date"])
        predicted_rates = pd.read_csv(predicted_rates_file, index_col = [0,1])
        predicted_rates["date"] = pd.to_datetime(predicted_rates["date"])
        predicted_rates_linear = pd.read_csv(predicted_linear_rates_file, index_col = [0,1])
        predicted_rates_linear["date"] = pd.to_datetime(predicted_rates_linear["date"])
        calibration_results = calibration_results.loc[~calibration_results['fomc_date'].isna()]
        calibration_results = calibration_results.loc[~calibration_results['alt_fomc_date'].isna()]

    # delta_observed_frame: the forward looking rate deltas for each frequency: 1bday, 5bday, 10bday.
    delta_observed_frame = get_rate_deltas(rates, calibration_results)

    # The results of Section 3 are produced by the following:
    do_expected_rates_analysis(rates, delta_observed_frame, predicted_rates, fomc_dates)

    # The results of Section 4 are produced by the following:
    do_dependency_analysis(delta_observed_frame, calibration_results, BUMP_UP_TRAINING_HORIZON - BUMP_DOWN_TRAINING_HORIZON)

    # The results of Section 5 are produced by the following:
    do_rate_change_explanatory_analysis(effr_rates, delta_observed_frame, predicted_rates_linear, calibration_results)
    
    # The results of Section 6 are produced by the following:
    do_risk_case_studies(effr_rates, delta_observed_frame, predicted_rates_linear, calibration_results)
