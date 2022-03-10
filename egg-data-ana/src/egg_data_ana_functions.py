"""
Functions for analyzing data from the Egg Counter research system

Author: Cody Jarrett
Organization: Phillips Lab, Institute of Ecology and Evolution,
              University of Oregon

"""
import csv
import random
import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.graph_objects as go

from scipy import stats
from scipy.stats import t
from exp_mixture_model import EMM
from matplotlib.figure import Figure
from matplotlib.pyplot import figure


def access_csv(file_str: str, data: list, usage_str: str):
    """
    It takes a string that represents the file name, a list of lists that represents the data, and a
    string that represents the usage of the file. It then opens the file, writes the data to the file,
    and closes the file

    Args:
      file_str (str): the name of the file you want to write to.
      data (list): list of lists
      usage_str (str): "w" means write, "a" means append.
    """
    with open(file_str, usage_str, newline="") as file:
        writer = csv.writer(file, delimiter=",")
        for el in data:
            writer.writerow(el)


def add_anomaly_data(egg_data: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe of egg data, add a column for each anomaly metric, and populate it with the
    results of running the breakpoint analysis on each row

    Args:
      egg_data (pd.DataFrame): The dataframe containing the egg temperature data

    Returns:
      A dataframe with the following columns added:
        - SetTemp
        - PreSlope
        - PostSlope
        - Ratio
        - AnomFlag
        - AnomEpoch
        - RegularIntervals
    """
    anom_cols = [
        "SetTemp",
        "PreSlope",
        "PostSlope",
        "Ratio",
        "AnomFlag",
        "AnomEpoch",
        "RegularIntervals",
    ]

    df = pd.DataFrame(
        columns=anom_cols, data=np.empty(shape=(len(egg_data), len(anom_cols)))
    )

    for i, row in enumerate(egg_data.itertuples()):
        out_dict = run_breakpoint_analysis(row)
        out_dict["reg_intervals"] = str(out_dict["reg_intervals"])
        out_list = [val for key, val in out_dict.items()]
        df.loc[i] = out_list

    df = df.drop(["SetTemp"], axis=1)

    egg_data = pd.concat([egg_data, df], axis=1)

    return egg_data


def apply_percentile_cutoff(
    egg_data: pd.DataFrame,
    set_temps: list,
    qt: float,
) -> pd.DataFrame:
    """
    Given a dataframe of egg counts, a list of temperatures, and a quantile threshold,
    this function will return a dataframe of egg counts with the specified quantile removed

    Args:
      egg_data (pd.DataFrame): The dataframe containing the egg data
      set_temps (list): list of temperatures to apply the cutoff to
      qt (float): float

    Returns:
      A dataframe with the quantile removed data.
    """
    qt_rm_dfs = []
    for t in set_temps:
        df = egg_data[egg_data["SetTemp"] == t]
        qt_series = df["EggCount"].quantile([qt])
        cutoff_val = round(qt_series.iloc[0], 2)
        qt_rm_df = df[df["EggCount"] <= cutoff_val]

        num_worms = len(qt_rm_df)
        mean_eggs = round(qt_rm_df["EggCount"].mean(), 2)
        std_eggs = round(qt_rm_df["EggCount"].std(), 2)

        print(
            f"{t}: {num_worms} Worms, {mean_eggs} Mean Eggs, {std_eggs} STD Eggs"
        )
        print(f"{qt*100}% Quantile: {cutoff_val}\n")

        qt_rm_dfs.append(qt_rm_df)

    qt_rm_data = pd.concat(qt_rm_dfs)

    egg_data = egg_data.loc[qt_rm_data.index]

    egg_data = egg_data.reset_index(drop=True)

    return egg_data


def conv_str_series_data(str_series: str) -> list:
    """
    Given a list of intervals, estimate the parameters of the model

    :param str_series: the series of intervals
    :return: a list of three values: p, l_1, and l_2.
    """
    str_list = str_series.strip("[]").split(",")
    float_list = [float(x) for x in str_list]
    return float_list


def detect_anomaly(
    ratio: float, pre_eggs_list: list, post_eggs_list: list
) -> dict:
    """
    If the ratio of pre-eggs to post-eggs is less than 0.5 or greater than 2.0, then the decision is to
    keep the pre-eggs. If the ratio is between 0.5 and 2.0, then the decision is to keep neither. If the
    ratio is exactly 1.0, then the decision is to keep both

    Args:
        ratio (float): The ratio of the number of eggs in the pre-section to the number of eggs in the
    post-section.
        pre_eggs_list (list): list of the number of eggs in each section of the pre-eggs
        post_eggs_list (list): list of eggs detected in the post-hatch section

    Returns:
        A dictionary with the following keys:
        - decision: string, either "Keep Pre", "Keep Post", or "Keep Neither"
        - pre_keep_flag: boolean, True if pre_eggs_list should be kept
        - post_keep_flag: boolean, True if post_eggs_list should be kept
        - anom_flag
    """
    anomaly_flag = False
    decision = "Regular"

    if (ratio < 0.5) or (ratio > 2.0):
        anomaly_flag = True

    pre_keep_flag = True
    post_keep_flag = True
    if anomaly_flag:
        if len(pre_eggs_list) > len(post_eggs_list):
            post_keep_flag = False
            decision = "Keep Pre"
        elif len(post_eggs_list) > len(pre_eggs_list):
            pre_keep_flag = False
            decision = "Keep Post"
        # If both sections have same num of eggs
        else:
            pre_keep_flag = False
            post_keep_flag = False
            decision = "Keep Neither"

    return {
        "decision": decision,
        "pre_keep_flag": pre_keep_flag,
        "post_keep_flag": post_keep_flag,
        "anom_flag": anomaly_flag,
    }


def estimate_parameters(
    raw_intervals: pd.Series, iterations: int = 100
) -> list:
    """
    Given a series of intervals, estimate the parameters of the double-exponential distribution

    Args:
      raw_intervals (pd.Series): The raw intervals of the data.
      iterations (int): the number of iterations to run the EM algorithm for. Defaults to 100

    Returns:
      a list of 3 parameter estimates: p, l_1, l_2.
    """
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


def get_longest_normal_vec_info(
    sfes: list, eggs_list: list, line_vec: np.array, line_vec_norm: float
) -> tuple:
    """
    Given a list of egg times, a list of eggs, and a line vector,
    return the index of the point furthest from the line vector,
    and the projection of that point onto the line vector

    Args:
      sfes (list): list of seconds from experiment start values
      eggs_list (list): list of the egg numbers
      line_vec (np.array): the vector of the line
      line_vec_norm (float): the norm of the line vector

    Returns:
      The index of the point furthest from the line, and the projections of the points onto the line.
    """
    projs = []
    orth_norms = []
    for p_x, p_y in zip(sfes, eggs_list):
        curve_vec = np.array([p_x, p_y])
        proj = (np.dot(curve_vec, line_vec) / (line_vec_norm ** 2)) * line_vec
        projs.append(proj)
        orth_vec = curve_vec - proj
        orth_vec_norm = np.linalg.norm(orth_vec)
        orth_norms.append(orth_vec_norm)

    furthest_point_norm = max(orth_norms)
    idx = orth_norms.index(furthest_point_norm)

    return (idx, projs)


def get_param_error_arrays(
    file_str: str, param: str, est_df: pd.DataFrame, temps: list
) -> list:
    """
    Given a file string, a parameter, an estimates dataframe, and a list of temperatures,
    return two arrays, one for the upper error bars and one for the lower error bars.

    Args:
      file_str (str): the file name of the bootstrap data
      param (str): the parameter of interest
      est_df (pd.DataFrame): the dataframe of estimated parameters
      temps (list): list of temperatures

    Returns:
      List of lists with two arrays, one for the upper error bar and one for the lower error bar.
    """
    upper_qt = 0.975
    lower_qt = 0.025

    bootstrap_df = pd.read_csv(file_str, header=0)

    array_plus = []
    array_minus = []

    # Build error bar arrays
    for t in temps:
        t_df = bootstrap_df[bootstrap_df["t"] == t]

        qt_series = t_df[param].quantile([lower_qt, upper_qt])

        lower_cutoff = round(qt_series.iloc[0], 4)
        upper_cutoff = round(qt_series.iloc[1], 4)

        param_est = est_df[est_df["Temperature"] == t][param]

        upper_offset = round(float(upper_cutoff - param_est), 4)
        lower_offset = round(float(param_est - lower_cutoff), 4)

        array_plus.append(upper_offset)
        array_minus.append(lower_offset)

    return [array_plus, array_minus]


def get_pool_estimates(set_temps: list, pooled_dict: list) -> dict:
    """
    Given a list of temperatures and a dictionary of intervals,
    return a dictionary of estimates for the temperatures

    Args:
      set_temps (list): list of temperatures to estimate parameters for
      pooled_dict (list): a dictionary of pd.Series, where the keys are temperatures and the
    values are pd.Series of egg lay interval data

    Returns:
      A dictionary with temperatures as keys and a list of parameter estimates as values.
    """
    est_dict = dict.fromkeys(set_temps, [])

    for t, intervals in pooled_dict.items():
        print(f"Estimating parameters for {t}, {len(intervals)} intervals")
        estimates = estimate_parameters(intervals)
        est_dict[t] = estimates

    return est_dict


def get_regular_intervals(
    pre_sfes: list,
    post_sfes: list,
    pre_keep_flag: bool,
    post_keep_flag: bool,
) -> list:
    """
    Calculates the intervals for the "regular" egg laying epoch. If pre_keep_flag,
    the "regular" epoch is the pre-breakpoint region. If post_keep_flag, the
    "regular" epoch is the post-breakpoint region. If both flags are True,
    the whole egg-laying trajectory is considered "regular".

    Args:
      pre_sfes (list): list of pre-sleep fes
      post_sfes (list): list of the sfes in the post-intervention period
      pre_keep_flag (bool): True if you want to include pre-SFES intervals in the regular intervals
      post_keep_flag (bool): True if you want to keep the post-SFES intervals

    Returns:
      A list of regular intervals.
    """
    reg_intervals = []

    if pre_keep_flag:
        pre_sfes_sec = [(x * 60 * 60) for x in pre_sfes]
        pre_intervals = np.diff(pre_sfes_sec, n=1)

        pre_intervals = normalize_tiny_intervals(pre_intervals)

        reg_intervals.extend(pre_intervals)

    if post_keep_flag:
        post_sfes_sec = [(x * 60 * 60) for x in post_sfes]
        post_intervals = np.diff(post_sfes_sec, n=1)

        post_intervals = normalize_tiny_intervals(post_intervals)

        reg_intervals.extend(post_intervals)

    return reg_intervals


def get_windowed_egg_counts(row: pd.Series) -> pd.DataFrame:
    """
    For each row,
    convert the SFES column into a series of datetime objects, and then
    convert that into a DataFrame with a time index. Then resample the
    DataFrame to 1 hour bins, and fill in missing values with the last
    known value. Then add a column with the time bin. Then group by the
    time bin and sum the values

    Args:
      row (pd.Series): pd.Series

    Returns:
      A dataframe with the time bins and the number of eggs laid in each bin.
    """
    # Get date (for helping keep track of the relative times)
    date = row.Date

    # Get SFES series for that row/worm
    sfes = conv_str_series_data(row.SFES)

    # Set a fake egg lay event at the end of the experiment time period
    # to help Pandas resample the time correctly. That way, I don't have
    # to do custom time filling.
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

    # Convert SFES series into a series of datetime objects
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


def load_ec_log(
    file_path: str,
    bad_rows: list,
    wrong_temp_sets: list,
    min_eggs_num: int = 10,
) -> pd.DataFrame:
    """
    Loads the egg counter data excel file, slices out the extra header rows, drops text cols,
    filters out worms with too few eggs, removes rows that somehow break everything in the
    breakpoint analysis, and adds an Experiment column

    Args:
      file_path (str): The path to the excel file.
      bad_rows (list): list
      wrong_temp_sets (list): list = [
      min_eggs_num (int): int = 10. Defaults to 10

    Returns:
      A dataframe with the egg counter log columns, and an added Experiment column
    """
    # Load all worksheets (all temperatures) with sheet_name=None
    data_dict = pd.read_excel(
        file_path, header=0, sheet_name=None, engine="openpyxl"
    )

    # Concat the returned dictionary of DataFrames
    data = pd.concat(data_dict)

    # Reset the index to collapse the multi-index to a single
    data = data.reset_index(drop=True)

    # Slice out the extra header rows
    data = data[data["Rig"] != "Rig"]

    # Drop text cols
    data = data.drop(["TempDataQuality", "Notes"], axis=1)

    # Get only entries that had eggs and non-zero params
    # (which also removes most nans)
    data = data[data["p"] > 0]

    # Filter out worms with too few eggs
    data = data[data["EggCount"] >= min_eggs_num]
    data = data.reset_index(drop=True)

    # Remove rows that somehow break everything in the breakpoint analysis
    for r in bad_rows:
        index = data.index[
            (data["SetTemp"] == r[0])
            & (data["Rig"] == r[1])
            & (data["Date"] == r[2])
            & (data["Lane"] == r[3])
        ].tolist()

        data = data.drop(index)

    # Explicitly set params as floats
    data["p"] = data["p"].astype("float")
    data["lambda1"] = data["lambda1"].astype("float")
    data["lambda2"] = data["lambda2"].astype("float")

    # Change experiments that were set to one temperature, but ended up
    # actually being a different temperature.
    # Here, "temp" is the actual temperature, NOT the set temperature.
    # This finds the experiment rig and date, and sets the "SetTemp" column
    # value for that entry to the actual temperature in the "wrong_temp_sets"
    # config value.
    for el in wrong_temp_sets:
        rig = el[0]
        date = el[1]
        temp = el[2]
        data.loc[(data.Rig == rig) & (data.Date == date), "SetTemp"] = temp

    # Add experiment column to make graphing of trajectories colored by experiment easier
    data["Experiment"] = data["Date"] + "_" + data["Rig"]

    return data


def normalize_tiny_intervals(intervals: list) -> list:
    """
    Given a list of intervals, round each interval to the nearest integer and replace any zeros with
    ones

    Args:
      intervals (list): list of intervals to normalize

    Returns:
      A list of intervals.
    """
    intervals = [round(x, 0) for x in intervals]
    for i, item in enumerate(intervals):
        if item == 0.0:
            intervals[i] = 1.0
    return intervals


def plot_kde(
    intervals_pool_dict: dict,
    colors: list,
    title: str,
    save_pic_flags: dict,
    figs_dir: str,
    width: int = 750,
    height: int = 500,
):
    """
    Plot the kernel density estimation of the logarithm of the intervals

    Args:
      intervals_pool_dict (dict): dictionary of pooled intervals for each temperature
      colors (list): list of color hex codes
      title (str): plot title
      save_pic_flags (dict): flags that control whether to save pictures
      figs_dir (str): directory of where to save pictures
      width (int): figure width
      height (int): figure height
    """
    fig = go.Figure()

    for (t, intervals), c in zip(intervals_pool_dict.items(), colors):

        intervals = [np.log(x) for x in intervals]

        X = np.array(intervals)
        X = X.reshape(-1, 1)

        kde = sm.nonparametric.KDEUnivariate(X)
        kde.fit()

        fig.add_trace(
            go.Scatter(
                x=kde.support,
                y=kde.density,
                mode="lines",
                line_color=c,
                name=t,
            )
        )

    fig = setup_figure(fig, width=width, height=height, title=title)

    save_pics(fig, figs_dir, title, save_pic_flags)

    fig.show()


def plot_param_bar_graph(
    resamples_file_str: str,
    param: str,
    est_df: pd.DataFrame,
    temps: list,
    title: str,
    colors: list,
    save_pic_flags: dict,
    figs_dir: str,
    width: int = 1200,
    height: int = 600,
):
    """
    Given a file containing resampled interval data, a parameter to graph, a dataframe of estimates, a
    list of temperatures, a title, a list of colors, a dictionary of flags for saving the figures, and a
    directory for saving the figures, plot the parameter estimates as a bar graph with error bars.

    Args:
      resamples_file_str (str): path to the resampled interval data file
      param (str): the parameter to graph
      est_df (pd.DataFrame): the dataframe containing the estimated parameters
      temps (list): list of temperatures
      title (str): figure title
      colors (list): list of colors for the bars
      save_pic_flags (dict): flags that control whether to save pictures
      figs_dir (str): directory of where to save pictures
      width (int): figure width, defaults to 1200
      height (int): figure height, defaults to 600
    """
    err_arrays = get_param_error_arrays(
        resamples_file_str, param, est_df, temps
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=temps,
            y=est_df[param],
            marker_color=colors,
            text=est_df[param],
            error_y=dict(
                type="data",
                symmetric=False,
                array=err_arrays[0],
                arrayminus=err_arrays[1],
            ),
        )
    )

    fig.update_traces(textposition="outside")

    fig = setup_figure(
        fig, width=width, height=height, show_legend=False, title=title
    )

    save_pics(fig, figs_dir, title, save_pic_flags)

    fig.show()


def pool_intervals(
    set_temps: list, t_dfs: list, reg_epoch_flag: bool = False
) -> dict:
    """
    Given a list of temperatures and a list of dataframes,
    the function returns a dictionary of pandas series,
    where the keys are the temperatures and the values are the series of intervals

    Args:
      set_temps (list): list of temperatures for which you want to pool intervals
      t_dfs (list): list of dataframes
      reg_epoch_flag (bool): If True, then the regular epochs are used. Defaults to False

    Returns:
      A dictionary of pd.Series objects.
    """
    pooled_dict = {}

    for t, df in zip(set_temps, t_dfs):
        temp_list = []

        for row in df.itertuples():

            if reg_epoch_flag:
                intervals_str = row.RegularIntervals
                intervals = conv_str_series_data(intervals_str)
            else:
                intervals_str = row.Intervals
                intervals = conv_str_series_data(intervals_str)
                intervals = normalize_tiny_intervals(intervals)

            temp_list.extend(intervals)

        pooled_dict[t] = pd.Series(temp_list)

    return pooled_dict


def randomized_parameter_test(
    egg_data: pd.DataFrame,
    param: str,
    t1: str,
    t2: str,
    plot_settings: dict,
    permutation_total: int = 1000,
    plot_stuff: bool = False,
    verbose: bool = False,
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


def run_breakpoint_analysis(
    row: pd.Series,
    plot_analysis: bool = False,
    title: str = "",
    figs_dir: str = "",
    save_pic_flags: dict = {},
) -> dict:

    t = row.SetTemp

    sfes = conv_str_series_data(row.SFES)
    sfes = [(x - sfes[0]) for x in sfes]
    sfes = [(x * (1 / 60) * (1 / 60)) for x in sfes]

    eggs_list = list(range(len(sfes)))

    line_vec = np.array([sfes[-1], eggs_list[-1]])
    line_vec_norm = np.linalg.norm(line_vec)

    idx, projs = get_longest_normal_vec_info(
        sfes, eggs_list, line_vec, line_vec_norm
    )

    ov_x1, ov_y1 = sfes[idx], eggs_list[idx]
    ov_x2, ov_y2 = projs[idx]

    pre_sfes = sfes[: sfes.index(ov_x1) + 1]
    post_sfes = sfes[sfes.index(ov_x1) :]

    pre_eggs_list = eggs_list[: eggs_list.index(ov_y1) + 1]
    post_eggs_list = eggs_list[eggs_list.index(ov_y1) :]

    pre_reg = stats.linregress(pre_sfes, pre_eggs_list)
    post_reg = stats.linregress(post_sfes, post_eggs_list)

    pre_s = round(pre_reg.slope, 4)
    post_s = round(post_reg.slope, 4)

    ratio = round(post_s / pre_s, 4)

    anomaly_res = detect_anomaly(ratio, pre_eggs_list, post_eggs_list)

    decision = anomaly_res["decision"]
    pre_keep_flag = anomaly_res["pre_keep_flag"]
    post_keep_flag = anomaly_res["post_keep_flag"]
    anomaly_flag = anomaly_res["anom_flag"]

    anom_epoch = "Neither"
    if decision == "Keep Pre":
        anom_epoch = "Post"
    if decision == "Keep Post":
        anom_epoch = "Pre"

    reg_intervals = get_regular_intervals(
        pre_sfes, post_sfes, pre_keep_flag, post_keep_flag
    )

    pre_line_y = [((pre_reg.slope * x) + pre_reg.intercept) for x in pre_sfes]
    post_line_y = [
        ((post_reg.slope * x) + post_reg.intercept) for x in post_sfes
    ]

    if plot_analysis:
        fig = go.Figure()

        # Pre and post regions
        fig.add_trace(
            go.Scatter(
                x=pre_sfes, y=pre_eggs_list, mode="lines", name="Pre-Epoch"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=post_sfes, y=post_eggs_list, mode="lines", name="Post-Epoch"
            )
        )

        # Orthogonal vector line
        fig.add_trace(
            go.Scatter(
                x=[ov_x1, ov_x2],
                y=[ov_y1, ov_y2],
                mode="lines",
                name="Orthogonal Vector",
            )
        )

        # First to last point line
        fig.add_trace(
            go.Scatter(
                x=[0, line_vec[0]],
                y=[0, line_vec[1]],
                mode="lines",
                name="First to Last Point",
                line=dict(dash="dash"),
            )
        )

        # Pre-epoch linear fit
        fig.add_trace(
            go.Scatter(
                x=pre_sfes,
                y=pre_line_y,
                mode="lines",
                name="Pre-Epoch Linear Fit",
                line=dict(dash="dot"),
            )
        )

        # Post-epoch linear fit
        fig.add_trace(
            go.Scatter(
                x=post_sfes,
                y=post_line_y,
                mode="lines",
                name="Post-Epoch Linear Fit",
                line=dict(dash="dot"),
            )
        )

        title = f"{title} - Ratio {ratio} - {decision}"
        x_title = "Time (hours)"
        y_title = "Egg Number"

        fig = setup_figure(
            fig,
            width=1000,
            height=600,
            show_legend=True,
            title=title,
            x_title=x_title,
            y_title=y_title,
        )

        fig.update_xaxes(range=[-1, 48])

        if save_pic_flags and figs_dir != "":
            save_pics(fig, figs_dir, title, save_pic_flags)
        else:
            print("Did not save pictures of plot")

        fig.show()

    return {
        "t": t,
        "pre_s": pre_s,
        "post_s": post_s,
        "ratio": ratio,
        "anom_flag": anomaly_flag,
        "anom_epoch": anom_epoch,
        "reg_intervals": reg_intervals,
    }


def save_pics(fig: Figure, figs_dir: str, title: str, save_pic_flags: dict):
    """
    Save a figure to files

    Args:
      fig (Figure): Figure
      figs_dir (str): The directory where the figures will be saved
      title (str): The title of the plot
      save_pic_flags (dict): dict
    """
    if save_pic_flags["png"]:
        fig.write_image(f"{figs_dir}/PNGs/{title}.png")

    if save_pic_flags["svg"]:
        fig.write_image(f"{figs_dir}/SVGs/{title}.svg", engine="kaleido")


def setup_figure(
    fig: Figure,
    width: int,
    height: int,
    show_legend: bool = True,
    title: str = "",
    x_title: str = "",
    y_title: str = "",
) -> Figure:
    fig.update_layout(
        width=width,
        height=height,
        showlegend=show_legend,
        title=title,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        plot_bgcolor="white",
    )

    fig.update_xaxes(
        title_text=x_title,
        showline=True,
        linewidth=2,
        linecolor="black",
        zeroline=False,
    )

    fig.update_yaxes(
        title_text=y_title,
        showline=True,
        linewidth=2,
        linecolor="black",
        zeroline=False,
    )

    return fig


def setup_pooled_estimates_for_graphing(
    est_dict: dict, params: list
) -> pd.DataFrame:
    """
    This function takes a dictionary of estimates and a list of parameters and returns a dataframe of
    estimates

    Args:
      est_dict (dict): a dictionary of the estimates
      params (list): list = ['p', 'lambda1', 'lambda2']

    Returns:
      A dataframe with the following columns:
        - Temperature
        - Estimate for each parameter
    """
    est_df = pd.DataFrame(est_dict)
    est_df = est_df.T
    est_df.columns = params
    est_df["Temperature"] = est_df.index

    return est_df
