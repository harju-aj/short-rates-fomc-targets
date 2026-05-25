
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

def get_factor_sensitivities(parameters, dates, horizon, jump, days_to_maturity, days_to_fomc_column):
    
    parameters_this_tenor = parameters.copy()
    parameters_this_tenor = parameters_this_tenor.loc[parameters_this_tenor["date"].isin(dates)]
    parameters_this_tenor["loading"] = (days_to_maturity + horizon - (parameters_this_tenor[days_to_fomc_column] + 1)) / days_to_maturity
    parameters_this_tenor.loc[parameters_this_tenor["loading"] < 0, "loading"] = 0
    sensitivities_pivot = parameters_this_tenor.pivot(index = 'date', columns = 'fomc')["loading"]
    sensitivities_pivot = sensitivities_pivot.reset_index()
    sensitivities_pivot.columns = sensitivities_pivot.columns.rename(None)
    sensitivities_pivot["delta_effr"] = 1.0
    if jump:
        sensitivities_pivot[0] = 1.0
    n_factors = sensitivities_pivot.shape[1] - 2
    factor_names = ["delta_effr"] + list(np.arange(n_factors))
    sensitivities_pivot = sensitivities_pivot[["date"] + factor_names]
    sensitivities_pivot = sensitivities_pivot.fillna(0)
    return sensitivities_pivot

# Get the risk factors for the periods that have a FOMC meeting:
def get_risk_factors_one_horizon_jump(parameters, effrs, dates, n_fomcs, fomc_dates_column):
    
    effr_dates = list(effrs["date"].values)
    effr_values = list(effrs["effr"].values)
    unique_fomc_dates = parameters[fomc_dates_column].unique()
    unique_fomc_dates = np.sort(unique_fomc_dates)
    last_date = np.max(dates)
    unique_fomc_dates = [pd.to_datetime(x) for x in unique_fomc_dates if pd.to_datetime(x) < last_date]
    factor_dates = []
    # effr_changes_over_full_horizon has the actual effr changes over the entire time window (such as 5 days)
    # The manuscript uses the notation r_{0t}(x) - r_0 for the elements in this vector. 
    effr_changes_over_full_horizon = []
    # factor_values has the changes in the predicted jumps for the upcoming meetings, but not including the one that takes place in the time window
    factor_values = []
    # the dates with last effr quote before the FOMC meeting: 
    effr_quote_dates_before_fomcs  = []
    for x in unique_fomc_dates:
        # this_factor_date is the factor date for the period that contains the fomc date (the fomc date may be the factor date itself, so the period is left-close-and-right-open)
        # in either case, the jump resulting from FOMC meeting happens during the period that starts on this_factor_date
        this_factor_date = [xx for xx, yy in zip(dates[0:-1], dates[1:]) if (x >= xx  and x < yy)]
        this_factor_date = this_factor_date[0]
        # similar for the effr quote dates: 
        this_effr_date = [xx for xx, yy in zip(effr_dates[0:-1], effr_dates[1:]) if (x >= xx  and x < yy)]
        this_effr_date = this_effr_date[0]        
        factor_dates.append(this_factor_date)
        effr_quote_dates_before_fomcs.append(this_effr_date)
        next_factor_date_index = list(dates).index(this_factor_date) + 1
        next_factor_date = dates[next_factor_date_index]
        this_parameters = parameters.loc[parameters["date"] == this_factor_date]
        this_parameters = this_parameters.sort_values("fomc")
        next_parameters = parameters.loc[parameters["date"] == next_factor_date]
        next_parameters = next_parameters.sort_values("fomc")
        this_jumps = this_parameters["jump"].values[1:]    # The jump where fomc == 0 is handled independently below
        next_jumps = next_parameters["jump"]               # In the horizon, the jump associated with fomc == i is the jump associated with fomc == i+1 previous date. 
        this_effr = this_parameters.iloc[0].effr
        next_effr = next_parameters.iloc[0].effr
        effr_changes_over_full_horizon.append((next_effr - this_effr))
        this_factors = [xx - yy for xx,yy in zip(next_jumps, this_jumps)]
        while len(this_factors) < n_fomcs - 1: 
            this_factors += [0]
        factor_values.append(this_factors)

    jump_deltas = pd.DataFrame(index = factor_dates, data = factor_values).reset_index()
    jump_deltas.columns = ["date"] + [str(x) for x in np.arange(1, n_fomcs)]

    unique_fomc_indexes = [effr_dates.index(x) for x in effr_quote_dates_before_fomcs] 
    unique_fomc_index_plus_one = [x + 1 for x in unique_fomc_indexes]
    
    # observed_jumps are the jumps that take place right after the FOMC meeting. 
    # The manuscript denotes these as tilde(j)_{1t}(x)
    observed_jumps = [effr_values[x] - effr_values[y] for x,y in zip(unique_fomc_index_plus_one, unique_fomc_indexes)]
    observed_jumps = pd.DataFrame({"date": factor_dates, fomc_dates_column: unique_fomc_dates, "observed_jump": observed_jumps, "effr_changes_over_horizon": effr_changes_over_full_horizon})
    
    # The columnd delta_effr has the residual factors (residual after the jump is controlled), denoted by: tilde(r)_{0t}(x) - r_{0t}
    observed_jumps["delta_effr"] = observed_jumps["effr_changes_over_horizon"] - observed_jumps["observed_jump"]
    jumps = parameters.loc[parameters["fomc"] == 0]
    jumps = jumps[["date", "jump"]]
    factors = jumps.merge(observed_jumps, on = 'date')

    # The zero'th factor is denoted by tilde(j)_{1t}(x) - j_{1t}(x) in the manuscript 
    factors["0"] = factors["observed_jump"] - factors["jump"]
    factors = factors.merge(jump_deltas, on = 'date')
    factors = factors.drop(["jump", fomc_dates_column, "observed_jump", "effr_changes_over_horizon"], axis = 1)
    return factors

# Risk factors for the periods that dont have FOMC meetings: 
def get_risk_factors_one_horizon_nojump(parameters, dates, fomc_dates_column):
    
    unique_fomcs = parameters["fomc"].unique()
    unique_fomcs.sort()
    risk_dates_parameters = parameters.loc[parameters["date"].isin(dates)]
    risk_dates_parameters = risk_dates_parameters.sort_values(["date", "fomc"])
    
    # fomc 1: 
    risk_dates_parameters_first_fomc = risk_dates_parameters.loc[risk_dates_parameters["fomc"] == 0]
    risk_dates_parameters_effr_pivot = risk_dates_parameters_first_fomc.pivot(index = 'date', columns =  fomc_dates_column)["effr"]
    risk_dates_parameters_effr_pivot = risk_dates_parameters_effr_pivot.sort_values("date")
    delta_effrs = risk_dates_parameters_effr_pivot.shift(-1) - risk_dates_parameters_effr_pivot
    # delta_effrs = delta_effrs.stack().reset_index().drop(fomc_dates_column, axis = 1)
    delta_effrs = delta_effrs.stack().dropna().reset_index().drop(fomc_dates_column, axis=1)
    delta_effrs.columns = ["date", "delta_effr"]
    
    risk_dates_parameters_jump_pivot = risk_dates_parameters_first_fomc.pivot(index = 'date', columns = fomc_dates_column)["jump"]
    delta_jumps = risk_dates_parameters_jump_pivot.shift(-1) - risk_dates_parameters_jump_pivot
    # delta_jumps = delta_jumps.stack().dropna().reset_index().drop(fomc_dates_column, axis = 1)
    delta_jumps = delta_jumps.stack().dropna().reset_index().drop(fomc_dates_column, axis=1)
    delta_jumps.columns = ["date", unique_fomcs[0]]

    factors = delta_effrs.merge(delta_jumps, on = ['date'])
    factor_columns = ["delta_effr", unique_fomcs[0]]
    # fomc > 1: 
    for x in unique_fomcs[1:]:
        risk_dates_parameters_this_fomc = risk_dates_parameters.loc[risk_dates_parameters["fomc"] == x]
        risk_dates_parameters_jump_pivot = risk_dates_parameters_this_fomc.pivot(index = 'date', columns = fomc_dates_column)["jump"]
        delta_jumps = risk_dates_parameters_jump_pivot.shift(-1) - risk_dates_parameters_jump_pivot
        # delta_jumps = delta_jumps.stack().dropna().reset_index().drop(fomc_dates_column, axis = 1)
        delta_jumps = delta_jumps.stack().dropna().reset_index().drop(fomc_dates_column, axis = 1)

        delta_jumps.columns = ["date", x]
        factors = factors.merge(delta_jumps, on = ['date'], how = 'outer')
        factor_columns.append(x)

    factors = factors[["date"] + factor_columns]
    factors = factors.fillna(0)
    return factors

def get_risk_factors_one_horizon(parameters, delta_observed, effrs, horizon, fomc_dates_column, days_to_fomc_column):

    fomcs = parameters["fomc"].unique()
    n_fomcs = len(fomcs)
    tenor_names = ["1mo", "3mo", "6mo"]
    days_to_maturities = [30, 90, 180]
    dates = delta_observed["date"].unique()
    dates = np.sort(dates)
    jump_sensitivities = {}
     
    jump_factors = get_risk_factors_one_horizon_jump(parameters, effrs, dates, n_fomcs, fomc_dates_column)
    jump_factor_dates = jump_factors["date"].unique()
    for x,y in zip(tenor_names, days_to_maturities):
        jump_sensitivities[x] = get_factor_sensitivities(parameters, jump_factor_dates, horizon, True, y, days_to_fomc_column)

    nojump_factors = get_risk_factors_one_horizon_nojump(parameters, dates, fomc_dates_column)
    nojump_factor_dates = nojump_factors["date"].unique()
    nojump_sensitivities = {}
    for x,y in zip(tenor_names, days_to_maturities):
        nojump_sensitivities[x] = get_factor_sensitivities(parameters, nojump_factor_dates, horizon, False, y, days_to_fomc_column)

    return jump_factors, nojump_factors, jump_sensitivities, nojump_sensitivities

def get_factor_residual_one_horizon(deltas, predicted_rates_linear, jump_factors, nojump_factors, jump_sensitivities, nojump_sensitivities, horizon):

    delta_column = "delta_" + str(horizon) + "d"
    roll_column = "rate_roll_" + str(horizon) + "d"
    tenor_names = ["1mo", "3mo", "6mo"]
    factor_names = list(jump_factors.columns[1:])

    # In the following we have to merge factors with their sensitivites. To avoid duplicates in column names, lets name the sensitivity columns as follows:
    sensitivity_columns = [str(x) + "_sensitivity" for x in factor_names]

    jump_residuals = {}
    nojump_residuals = {}
    for x in tenor_names:
        deltas_this_tenor = deltas.loc[deltas["tenor"] == x]
        predicted_rates_this_tenor = predicted_rates_linear.loc["1", x][["date", "rate", roll_column]].reset_index().drop(["model", "tenor"], axis = 1)   
        # The mean column in predicted_rates_this_tenor has the expected rates (or 'rolled') in the model:
        predicted_rates_this_tenor["mean"] = predicted_rates_this_tenor[roll_column] - predicted_rates_this_tenor["rate"]
        deltas_this_tenor = deltas_this_tenor.merge(predicted_rates_this_tenor, on = ["date"])
        
        jump_sensitivities_this_tenor = jump_sensitivities[x]
        jump_sensitivities_this_tenor.columns = ["date"] + sensitivity_columns
        jump_factors_this_tenor = jump_factors.merge(jump_sensitivities_this_tenor, on = 'date')
        # The factor_delta column is the component of delta that can be attributed to the factor changes: 
        jump_factors_this_tenor["factor_delta"] = 0
        for xx, yy in zip(factor_names, sensitivity_columns):
            jump_factors_this_tenor["factor_delta"] += jump_factors_this_tenor[xx] * jump_factors_this_tenor[yy]
        jump_factors_this_tenor = jump_factors_this_tenor[["date", "factor_delta"]]
        jump_deltas_this_tenor = deltas_this_tenor.merge(jump_factors_this_tenor, on = 'date')
        jump_deltas_this_tenor["risky_delta"] = jump_deltas_this_tenor[delta_column] - jump_deltas_this_tenor["mean"]
        jump_deltas_this_tenor["factor_residual"] = jump_deltas_this_tenor["risky_delta"] - jump_deltas_this_tenor["factor_delta"]
        ## 
        nojump_sensitivities_this_tenor = nojump_sensitivities[x]
        n1 = nojump_sensitivities_this_tenor.shape[1]
        n2 = 1 + len(sensitivity_columns)
        if n1 < n2:
            for i in range(n2- n1):
                nojump_sensitivities_this_tenor[f"pad_{i}"] = 0.0

        nojump_sensitivities_this_tenor.columns = ["date"] + sensitivity_columns
        nojump_factors_this_tenor = nojump_factors.merge(nojump_sensitivities_this_tenor, on = 'date')
        nojump_factors_this_tenor["factor_delta"] = 0
        for xx, yy in zip(nojump_factors_this_tenor.columns[1:], sensitivity_columns):
            nojump_factors_this_tenor["factor_delta"] += nojump_factors_this_tenor[xx] * nojump_factors_this_tenor[yy]
        nojump_factors_this_tenor = nojump_factors_this_tenor[["date", "factor_delta"]]
        nojump_deltas_this_tenor = deltas_this_tenor.merge(nojump_factors_this_tenor, on = 'date')
        nojump_deltas_this_tenor["risky_delta"] = nojump_deltas_this_tenor[delta_column] - nojump_deltas_this_tenor["mean"]
        nojump_deltas_this_tenor["factor_residual"] = nojump_deltas_this_tenor["risky_delta"] - nojump_deltas_this_tenor["factor_delta"]

        jump_deltas_this_tenor = jump_deltas_this_tenor[["date", delta_column, "risky_delta", "factor_residual", "factor_delta"]]
        nojump_deltas_this_tenor = nojump_deltas_this_tenor[["date", delta_column, "risky_delta", "factor_residual", "factor_delta"]]

        jump_residuals[x] = jump_deltas_this_tenor
        nojump_residuals[x] = nojump_deltas_this_tenor

    return jump_residuals, nojump_residuals

def fit_factor_residual_level_changes_helper(residuals_dict, tenor_names):
    all_residuals =  residuals_dict[tenor_names[0]].copy()
    all_residuals = all_residuals[["date", "factor_residual"]]
    all_residuals.columns = ["date", tenor_names[0]]
    for x in tenor_names[1:]:
        this_residuals = residuals_dict[x].copy()
        this_residuals = this_residuals[["date", "factor_residual"]]
        this_residuals.columns = ["date", x]
        all_residuals = all_residuals.merge(this_residuals, on = 'date')
    all_residuals["level_factor"] = all_residuals.iloc[:, 1:].mean(axis=1)

    for x in tenor_names:
        residuals_dict[x] = residuals_dict[x].merge(all_residuals[["date", "level_factor"]])
        residuals_dict[x]["residual"] = residuals_dict[x]["factor_residual"]  - residuals_dict[x]["level_factor"]

    return residuals_dict

def fit_factor_residual_level_changes(jump_residuals, nojump_residuals): 
    
    tenor_names = ["1mo", "3mo", "6mo"]
    jump_residuals = fit_factor_residual_level_changes_helper(jump_residuals, tenor_names)
    nojump_residuals = fit_factor_residual_level_changes_helper(nojump_residuals, tenor_names)
    all_residuals = {}
    for x in tenor_names:
        jump_residuals[x]["jump"] = 1
        nojump_residuals[x]["jump"] = 0
        all_residuals[x] = pd.concat([jump_residuals[x], nojump_residuals[x]])
        all_residuals[x] = all_residuals[x].sort_values("date")
    return all_residuals


def get_factor_risk_breakdown(jump_factors, nojump_factors, jump_sensitivities, nojump_sensitivities, all_residuals, date, window):
    
    date = pd.to_datetime(date)
    tenors = ["1mo", "3mo", "6mo"]
    delta_column = all_residuals["1mo"].columns[1+1]
    all_factor_contributions = pd.DataFrame()
    if date in jump_sensitivities["1mo"]["date"].values: 
        for x in tenors: 
            this_sensitivities = jump_sensitivities[x].loc[jump_sensitivities[x]["date"] == date]
            this_sensitivities = this_sensitivities.drop("date", axis = 1).values
            this_factors = jump_factors.loc[(jump_factors["date"] < date) & (jump_factors["date"] >= date - BDay(window))]
            this_level_factors = all_residuals[x][["date", "level_factor", "residual", delta_column]]
            this_factors = this_factors.merge(this_level_factors, on = 'date')
            this_residuals =  this_factors["residual"].values
            this_factors = this_factors.drop(["date","residual", delta_column], axis = 1)
            this_residual_sigma = this_residuals.std()            
            cov_matrix = this_factors.cov().values            
            sigma = np.sqrt(np.matmul(np.matmul(this_sensitivities, cov_matrix), this_sensitivities.T)[0][0] + this_residual_sigma ** 2 )
            X = np.matmul(cov_matrix, this_sensitivities.T)
            X = X.T[0] 
            factor_contributions = (this_sensitivities * X) / sigma            
            factor_contributions_frame = pd.DataFrame(columns = this_factors.columns, data = factor_contributions)
            factor_contributions_frame["tenor"] = x
            factor_contributions_frame = factor_contributions_frame[["tenor"] + list(this_factors.columns)]
            factor_contributions_frame["residual"] = this_residual_sigma
            factor_contributions_frame["sigma"] = factor_contributions_frame.drop("tenor", axis = 1).sum(axis = 1)
            all_factor_contributions = pd.concat([all_factor_contributions, factor_contributions_frame])
    
    if date in nojump_sensitivities["1mo"]["date"].values:
        for x in tenors: 
            this_sensitivities = nojump_sensitivities[x].loc[nojump_sensitivities[x]["date"] == date]
            this_sensitivities = this_sensitivities.drop("date", axis = 1).values
            this_factors = nojump_factors.loc[(nojump_factors["date"] < date) & (nojump_factors["date"] >= date - BDay(window))]
            this_level_factors = all_residuals[x][["date", "level_factor", "residual", delta_column]]
            this_factors = this_factors.merge(this_level_factors, on = 'date')
            this_residuals =  this_factors["residual"].values
            this_factors = this_factors.drop(["date","residual", delta_column], axis = 1)
            cov_matrix = this_factors.cov().values
            this_residual_sigma = this_residuals.std()            
            sigma = np.sqrt(np.matmul(np.matmul(this_sensitivities, cov_matrix), this_sensitivities.T)[0][0] + (this_residual_sigma ** 2))
            X = np.matmul(cov_matrix, this_sensitivities.T)
            X = X.T[0] 
            factor_contributions = (this_sensitivities * X) / sigma            
            factor_contributions_frame = pd.DataFrame(columns = this_factors.columns, data = factor_contributions)
            factor_contributions_frame["tenor"] = x
            factor_contributions_frame = factor_contributions_frame[["tenor"] + list(this_factors.columns)]
            factor_contributions_frame["residual"] = this_residual_sigma
            factor_contributions_frame["sigma"] = factor_contributions_frame.drop("tenor", axis = 1).sum(axis=1)
            all_factor_contributions = pd.concat([all_factor_contributions, factor_contributions_frame])
    
    columns = all_factor_contributions.columns
    all_factor_contributions["date"] = date
    all_factor_contributions = all_factor_contributions[["date"] + list(columns)]
    all_factor_contributions = all_factor_contributions.rename(columns = {0 : "factor 1", 1 : "factor 2", 2 : "factor 3", 3 : "factor 4", 4 : "factor 5"})
    all_factor_contributions = all_factor_contributions.rename(columns = {"0" : "factor 1", "1" : "factor 2", "2" : "factor 3", "3" : "factor 4", "4" : "factor 5"})
    return all_factor_contributions