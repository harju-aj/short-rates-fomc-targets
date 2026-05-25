
import os
import pandas as pd
from pandas.tseries.offsets import BDay, Day

futures_codes_to_months = {
    'F' : '01', 
    'G' : '02', 
    'H' : '03', 
    'J' : '04',
    'K' : '05', 
    'M' : '06', 
    'N' : '07',
    'Q' : '08', 
    'U' : '09', 
    'V' : '10', 
    'X' : '11', 
    'Z' : '12'
}

# Frequency can be either
#   "BDay" = business day, or
#   "CDay" = calendar day.
def get_dates(first_date, last_date, frequency):

    first_date_dt = pd.to_datetime(first_date)
    last_date_dt = pd.to_datetime(last_date)
    all_dates = [first_date_dt]
    if frequency == "BDay":
        running_date = first_date_dt + BDay(1)
        while(running_date <= last_date_dt):
            all_dates.append(running_date)
            running_date = running_date + BDay(1)
    elif frequency == "CDay":
        running_date = first_date_dt + Day(1)
        while(running_date <= last_date_dt):
            all_dates.append(running_date)
            running_date = running_date + Day(1)
    all_dates = pd.DataFrame(all_dates, columns = ['date'])
    all_dates["horizon_date"] = all_dates.shift(-1)
    all_dates["time_delta"] = (all_dates["horizon_date"] - all_dates["date"]).dt.days
    return all_dates


def forward_fill_columns(input_frame, dates_frame, columns_to_fill, group_by_column = None): 
    limit_in_bdays = 20
    new_dates_frame = dates_frame.reset_index()[["date"]]
    new_dates_frame["date"] = pd.to_datetime(new_dates_frame["date"])
    input_frame["date"] = pd.to_datetime(input_frame["date"])
    if group_by_column:
        columns_to_fill = columns_to_fill + [group_by_column]
        output_frame = pd.DataFrame()
        unique_group_keys = input_frame[group_by_column].unique()
        for x in unique_group_keys: 
            this_input_frame = input_frame.loc[input_frame[group_by_column] == x]
            this_first_date = this_input_frame["date"].min()
            this_last_date = this_input_frame["date"].max()
            this_input_frame = new_dates_frame.merge(this_input_frame, on = 'date', how = 'left')
            this_input_frame = this_input_frame.sort_values("date")
            for xx in columns_to_fill: 
                this_input_frame[xx] = this_input_frame[xx].ffill(limit = limit_in_bdays)
            this_input_frame = this_input_frame.loc[(this_input_frame["date"] >= this_first_date) & (this_input_frame["date"] <= this_last_date)]
            output_frame = pd.concat([output_frame, this_input_frame])
        output_frame = output_frame.sort_values(["date", group_by_column])
        return output_frame
    else:
        this_first_date = input_frame["date"].min()
        this_last_date = input_frame["date"].max()
        ouput_frame = new_dates_frame.merge(input_frame, on = 'date', how = 'left')
        for x in columns_to_fill:
            ouput_frame[x] = ouput_frame[x].ffill(limit = limit_in_bdays)
        ouput_frame = ouput_frame.loc[(ouput_frame["date"] >= this_first_date) & (ouput_frame["date"] <= this_last_date)]
        return ouput_frame

def process_future_prices(futures_directory): 
    X = os.listdir(futures_directory)
    X = [os.path.join(futures_directory, x) for x in X]
    output = pd.DataFrame()
    for x in X:
        year, month = '', ''
        filename = os.path.basename(x)
        symbol = filename.split('_')[0]
        year = '20' + symbol[-2:]
        month = futures_codes_to_months[symbol[2]]
        skip_lines = 0
        with open(x) as f:
            first_line_first_bit = f.readline().strip('\n').split(',')[0]
            if first_line_first_bit[1:7] == 'Symbol': 
                skip_lines = 1   
        this_frame = pd.read_csv(x, skiprows=skip_lines, skipfooter=1, engine='python')
        this_frame = this_frame.rename(columns = {'Date Time': 'date', 'Time' : 'date', 'Last' : 'price', 'Close' : 'price'})
        this_frame = this_frame[["date", "price"]]
        this_frame["year"] = int(year)
        this_frame["month"] = int(month)
        this_frame["date"] = pd.to_datetime(this_frame["date"])
        output = pd.concat([output, this_frame])
    output = output[["date", "year", "month", "price"]]
    output["contract"] = output.apply(lambda x: str(x["year"]) + "-" + str(x["month"]), axis = 1)
    return output

def smooth_effrs(inputs, fomc_dates): 

    inputs = inputs.sort_values('date')
    dates = inputs["date"].values
    effrs = inputs["effr"].values
    effrs_smoothed = effrs.copy()
    fomcs = fomc_dates["date"].values
    i = 1
    for date in dates[1:]: 
        date = pd.to_datetime(date)
        next_date = date + BDay(1)
        prev_date = date - BDay(1)
        date_after_fomc = False 
        if prev_date in fomcs: 
            date_after_fomc = True
        this_month = date.month 
        next_month = next_date.month
        if (this_month != next_month) and (not date_after_fomc): 
            effrs_smoothed[i] = effrs[i-1]
        i = i + 1
    inputs["effr"] = effrs_smoothed
    return inputs
