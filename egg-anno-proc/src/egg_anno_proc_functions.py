"""
Author: Cody Jarrett
Organization: Phillips Lab, Institute of Ecology and Evolution,
              University of Oregon

"""
import csv
import random
import datetime
import numpy as np
import pandas as pd

# import plotly.io as pio
import plotly.graph_objects as go

# pio.kaleido.scope.mathjax = None

from exp_mixture_model import EMM


def access_csv(file_str, data, usage_str):
    with open(file_str, usage_str, newline="") as file:
        writer = csv.writer(file, delimiter=",")
        for el in data:
            writer.writerow(el)


def load_ec_log(file_path, wrong_temp_sets, min_eggs_num=10):
    # Load all worksheets (all temperatures) with sheet_name=None
    data_dict = pd.read_excel(
        file_path, header=0, sheet_name=None, engine="openpyxl"
    )

    # Concat the returned dictionary of DataFrames
    data = pd.concat(data_dict)

    # Reset the index to collapse the multi-index to a single
    data = data.reset_index(drop=True)

    # Slice out the extra header rows and drop big text cols
    data = data[data["Rig"] != "Rig"]
    data = data.drop(["TempDataQuality", "Notes"], axis=1)

    # Get only entries that had eggs and non-zero params
    # (which also removes most nans)
    data = data[data["p"] > 0]

    # Filter out worms with too few eggs
    data = data[data["EggCount"] >= min_eggs_num]
    data = data.reset_index(drop=True)

    # Explicitly set params as floats
    data["p"] = data["p"].astype("float")
    data["lambda1"] = data["lambda1"].astype("float")
    data["lambda2"] = data["lambda2"].astype("float")

    for el in wrong_temp_sets:
        rig = el[0]
        date = el[1]
        temp = el[2]
        data.loc[(data.Rig == rig) & (data.Date == date), "SetTemp"] = temp

    # Get rid of C character in set temps
    #     data['SetTemp'] = [x.strip('C') for x in data['SetTemp']]
    #     data['SetTemp'].astype(int)

    return data


def conv_str_series_data(str_series):
    str_list = str_series.strip("[]").split(",")
    float_list = [float(x) for x in str_list]
    return float_list


def estimate_parameters(raw_intervals, iterations=100):
    model = EMM(k=2, n_iter=iterations)
    pi, mu = model.fit(raw_intervals)

    try:
        pi_2 = model.pi[1]
        mu_1 = model.mu[0]
        mu_2 = model.mu[1]
        p = round(1 - (pi_2 * mu_1 * ((1 / mu_1) - (1 / mu_2))), 5)
        l_1 = round(1 / mu_1, 5)
        if l_1 > 0.9999:
            l_1 = 0.9999
        l_2 = round(1 / (p * mu_2), 5)
    except Exception:
        pi_2 = 0
        mu_1 = 0
        mu_2 = 0
        p = 0
        l_1 = 0
        l_2 = 0

    return [p, l_1, l_2]


def get_windowed_egg_counts(row):
    # Get date (for helping keep track of the relative times)
    date = row.Date
    # Get SFES series for that row/worm
    sfes = conv_str_series_data(row.SFES)
    # Set a fake egg lay event at the end of the experiment time period
    # to help Pandas resample the time correctly. That way, I don't have
    # to do custom time filling stuff.
    # 172,000 seconds = 48 hours, which is the length of the experiment.
    # When looking at binned times, if you bin by hour, this results in
    # the last 46 minutes and 40 seconds being potentially missed in the
    # bins. So instead of adding a final SFES value of 172,000, I add
    # 172,800 (46 min, 40 sec = 800 seconds) to even the 1 hour bins
    sfes.append(172_800)

    # Set up first time as a datetime object at 0
    first_time = "00:00:00"
    first_dt = date + " " + first_time
    first_dt = datetime.datetime.strptime(first_dt, "%Y-%m-%d %H:%M:%S")

    # Basically, convert SFES series into a series of datetime objects
    # that preserves the relative timing of egg lay events.
    # The absolute times do not correspond to when the egg lay
    # occurred.
    dts = [first_dt]
    for t in sfes:
        next_dt = first_dt + datetime.timedelta(seconds=t)
        dts.append(next_dt)

    # Set up a DataFrame from the SFES datetime objects
    df = pd.DataFrame(dts, columns=["time"])
    # Set the DataFrame index to the time column
    df = df.set_index("time", drop=True)
    # At each time point, there was 1 egg laid. So set the "value"
    # column to all ones
    df["value"] = np.ones(len(df), dtype=int)
    # Remove the one at timepoint 0, because no eggs had been laid yet
    df.iloc[0]["value"] = 0
    # Set the fake egg lay at the end to 0, to remove the fake
    df.iloc[-1]["value"] = 0

    # Resample
    dfrs = df.resample("1h").sum().ffill()

    # Add bins
    dfrs["TimeBin"] = pd.cut(dfrs.index, bins=dfrs.index, right=False)

    # Group bins to make final dataframe with correct time bins and values
    dfg = dfrs.groupby(["TimeBin"]).sum()

    return dfg


def get_param_error_arrays(file_str, param, estimates_df, set_temps):
    upper_qt = 0.975
    lower_qt = 0.025

    bootstrap_df = pd.read_csv(file_str, header=0)

    array_plus = []
    array_minus = []

    # Build error bar arrays
    for t in set_temps:
        t_df = bootstrap_df[bootstrap_df["t"] == t]

        qt_series = t_df[param].quantile([lower_qt, upper_qt])

        lower_cutoff = round(qt_series.iloc[0], 4)
        upper_cutoff = round(qt_series.iloc[1], 4)

        param_est = estimates_df[estimates_df["Temperature"] == t][param]

        upper_offset = round(float(upper_cutoff - param_est), 4)
        lower_offset = round(float(param_est - lower_cutoff), 4)

        array_plus.append(upper_offset)
        array_minus.append(lower_offset)

    return array_plus, array_minus


def randomized_parameter_test(
    egg_data,
    param,
    t1,
    t2,
    plot_settings,
    permutation_total=1000,
    plot_stuff=False,
    verbose=False,
):
    temps = [t1, t2]

    dfs = [egg_data[egg_data["SetTemp"] == t] for t in temps]

    means = [round(df[param].mean(), 4) for df in dfs]

    # Calculate "ground truth" -- actual, observed difference between means
    # of the two temperature param estimate values
    gT = round(means[0] - means[1], 4)

    # Get param estimate vals into single array for easier shuffling and
    # slicing
    x = pd.concat(dfs)
    x = x[param]
    x = np.array(x)

    # Get lengths of temperature DataFrames for slicing and printing info
    df1_len = len(dfs[0])
    df2_len = len(dfs[0])

    test_stats = np.zeros(permutation_total)
    # Do permutations test
    for i in range(permutation_total):
        random.shuffle(x)
        mean1 = np.average(x[:df1_len])
        mean2 = np.average(x[df1_len:])
        test_stats[i] = round(mean1 - mean2, 4)

    # Get p-value for hypothesis test
    p_val = round(len(np.where(test_stats >= gT)[0]) / permutation_total, 4)

    if verbose:
        print(f"{temps[0]} v {temps[1]} - {param}")
        print("===============")
        print(f"{temps[0]} Count: {df1_len}")
        print(f"{temps[1]} Count: {df2_len}")

        print(f"\n{temps[0]} Mean: {means[0]}")
        print(f"{temps[1]} Mean: {means[1]}")

        print(f"\nObserved {temps[0]} Mean - {temps[1]} Mean: {gT}")

        print(f"p-value: {p_val}")

    if plot_stuff:
        title = (
            f"Randomized Parameter Estimate Comparison Histogram -"
            f"{t1} v {t2} - {param} - gT {gT} - p-val {p_val}"
        )
        file_title = title.replace(" ", "")
        file_title = file_title.replace("-", "_")

        fig = go.Figure(data=[go.Histogram(x=test_stats, nbinsx=20)])

        fig.add_vline(x=gT, line_width=2, line_dash="dash", line_color="black")

        fig.update_layout(
            width=1000, height=600, showlegend=False, title_text=title
        )

        fig.update_yaxes(title_text="Frequency")

        if plot_settings["save_svg"]:
            svg_save_loc = plot_settings["svg_save_loc"]
            fig.write_image(
                f"{svg_save_loc}/Pairwise Randomized Tests/{file_title}.svg",
                engine="kaleido",
            )

        if plot_settings["save_png"]:
            png_save_loc = plot_settings["png_save_loc"]
            fig.write_image(
                f"{png_save_loc}/Pairwise Randomized Tests/{file_title}.png"
            )

        fig.show()

    return [df1_len, df2_len, means[0], means[1], gT, p_val]
