#!/usr/bin/env python3
# INSERT FILE DESCRIPTION

"""
Util functions to run Model. Includes Data loading, etc...
"""
import os
from typing import List, Union

import numpy as np
import pandas as pd

from tqdm import tqdm
import datetime as dt

tqdm.pandas()


def _compute_last_target_id(df: pd.DataFrame, time_col: str = "intime", mode: str = "max") -> pd.DataFrame:
    """Identify last ids given df according to time given by time_col column. Mode determines min or max."""
    if mode == "max":
        time = df[time_col].norm_max()
    elif mode == "min":
        time = df[time_col].norm_min()
    else:
        raise ValueError("mode must be one of ['min', 'max']. Got {}".format(mode))

    last_ids = df[df[time_col] == time]

    return last_ids


def _rows_are_in(df1: pd.DataFrame, df2: pd.DataFrame, matching_columns: Union[List[str], str]) -> pd.DataFrame:
    """
    Checks if values present in row of df1 exist for all columns in df2. Note that this does not necessarily mean
    the whole row of df1 is in df2, but is good enough for application.

    Returns: array of indices indicating the relevant rows of df1.
    """
    if isinstance(matching_columns, str):
        matching_columns = [matching_columns]

    # Iterate through each column
    matching_ids = np.ones(df1.shape[0])
    for col in tqdm(matching_columns):
        col_matching = df1[col].isin(df2[col].values).values  # where df1 col is subset of df2 col
        matching_ids = np.logical_and(matching_ids, col_matching)  # match with columns already looked at

    return matching_ids


def _compute_second_transfer_info(df: pd.DataFrame, time_col, target_cols):
    """
    Given transfer data for a unique id, compute the second transfer as given by time_col.

    return: pd.Series with corresponding second transfer info.
    """
    time_info = df[time_col]
    second_transfer_time = time_info[time_info != time_info.norm_min()].norm_min()

    # Identify second transfer info - can be empty, unique, or repeated instances
    second_transfer = df[df[time_col] == second_transfer_time]

    if second_transfer.empty:
        output = [df.name, df["hadm_id"].iloc[0], df["transfer_id"].iloc[0]] + [np.nan] * (len(target_cols) - 3)
        return pd.Series(data=output, index=target_cols)

    elif second_transfer.shape[0] == 1:
        return pd.Series(data=second_transfer.squeeze().values, index=target_cols)

    else:  # There should be NONE
        print(second_transfer)
        raise ValueError("Something's gone wrong! No expected repeated second transfers with the same time.")


def convert_columns_to_dt(df: pd.DataFrame, columns: Union[str, List[str]]):
    """Convert columns of dataframe to datetime format, as per given"""
    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        df[col] = pd.to_datetime(df.loc[:, col].values)

    return df


def subsetted_by(df1: pd.DataFrame, df2: pd.DataFrame, matching_columns: Union[List[str], str]) -> pd.DataFrame:
    """
    Subset df1 based on matching_columns, according to values existing in df2.

    Returns: pd.DataFrame subset of df1 for which rows are a subset of df2
    """

    return df1.iloc[_rows_are_in(df1, df2, matching_columns), :]


def endpoint_target_ids(df: pd.DataFrame, identifier: str, time_col: str = "intime", mode: str = "max") -> pd.DataFrame:
    """
    Given identifier target ("id"), compute the endpoint associated with time column.

    Returns: pd.DataFrame with ids and associated endpoint information.
    """
    last_ids = df.groupby(identifier, as_index=False).progress_apply(
        lambda x: _compute_last_target_id(x, time_col=time_col, mode=mode))

    return last_ids.reset_index(drop=True)


def compute_second_transfer(df: pd.DataFrame, identifier: str, time_col: str, target_cols: pd.Index) -> pd.DataFrame:
    """
    Given transfer data represented by unique identifier ("id"), compute the second transfer of the admission.
    Second Transfer defined as second present intime in the date (if multiple, this is flagged). If there are
    no transfers after, then return np.nan. target_cols is the target information columns.

    This function checks the second transfer intime is after outtime of first transfer record.

    Returns: pd.DataFrame with id and associated second transfer information (intime/outtime, unit, etc...)
    """
    second_transfer_info = df.groupby(identifier, as_index=False).progress_apply(
        lambda x: _compute_second_transfer_info(x, time_col, target_cols))

    return second_transfer_info.reset_index(drop=True)


def _has_many_nas(df: pd.DataFrame, targets: Union[List[str], str], min_count: int, min_frac: float) -> bool:
    """
    For a given admission/stay with corresponding vital sign information, return boolean indicating whether low
    missingness conditions are satisfied. These are:
    a) At least min_count observations.
    b) Proportion of missing values smaller than min_frac for ALL targets.

    returns: boolean indicating admission should be kept.
    """
    if isinstance(targets, str):
        targets = [targets]

    has_minimum_counts = df.shape[0] > min_count
    has_less_NA_than_frac = df[targets].isna().sum() <= min_frac * df.shape[0]

    return has_minimum_counts and has_less_NA_than_frac.all()


def remove_adms_high_missingness(df: pd.DataFrame, targets: Union[List[str], str],
                                 identifier: str, min_count: int, min_frac: float) -> pd.DataFrame:
    """
    Given vital sign data, remove admissions with too little information. This is defined as either:
    a) Number of observations smaller than allowed min_count.
    b) Proportion of missing values in ANY of the targets is higher than min_frac.

    Returns: pd.DataFrame - Vital sign data of the same type, except only admissions with enough information are kept.
    """
    output = df.groupby(identifier, as_index=False).filter(
        lambda x: _has_many_nas(x, targets, min_count, min_frac))

    return output.reset_index(drop=True)


def _resample_adm(df: pd.DataFrame, rule: str, time_id: str,
                  time_vars: Union[List[str], str], static_vars: Union[List[str], str]) -> pd.DataFrame:
    """
    For a particular stay with vital sign data as per df, resample trajectory data (subsetted to time_vars),
    according to index given by time_to_end and as defined by rule. It is important that time_to_end decreases
    throughout admissions and hits 0 at the end - this is for resampling purposes.

    Params:
    df: pd.Dataframe, containing trajectory and static data for each admission.
    rule: str, indicates the resampling rule (to be fed to pd.DataFrame.resample())

    static_vars is a list of relevant identifier information

    returns: Resampled admission data. Furthermore, two more info columns are indicated (chartmax and chartmin).
    """
    if isinstance(time_vars, str):
        time_vars = [time_vars]

    if isinstance(static_vars, str):
        static_vars = [static_vars]

    # Add fake observation (with missing values) so that resampling starts at end of admission
    df_inter = df[time_vars + ["time_to_end"]]
    df_inter = df_inter.append(pd.Series(data=[np.nan] * len(time_vars) + [dt.timedelta(seconds=0)],
                                         index=df_inter.columns), ignore_index=True)

    # resample on time_to_end axis
    output = df_inter.sort_values(by="time_to_end", ascending=False).resample(
        on="time_to_end",
        rule=rule, closed="left", label="left").mean()

    # Compute static ids manually and add information about max and min time id values
    output[static_vars] = df[static_vars].iloc[0, :].values
    output[time_id + "_min"] = df[time_id].norm_min()
    output[time_id + "_max"] = df[time_id].norm_max()

    # Reset index to obtain resampled values
    output.index.name = f"sampled_time_to_end({rule})"
    output.reset_index(drop=False, inplace=True)

    return output


def compute_time_to_end(df: pd.DataFrame, id_key: str, time_id: str, end_col: str):
    """
    Compute time to end of admission for a given observation associated with a particular admission id.

    df: pd.DataFrame with trajectory information.
    id_key: str - column of df representing the unique id admission identifier.
    time_id: str - column of df indicating time observations was taken.
    end_col: str - column of df indicating, for each observation, the end time of the corresponding admission.

    returns: sorted pd.DataFrame with an extra column indicating time to end of admission. This will be used for
    resampling.
    """
    df_inter = df.copy()
    df_inter["time_to_end"] = df_inter[end_col] - df_inter[time_id]
    df_inter.sort_values(by=[id_key, "time_to_end"], ascending=[True, False], inplace=True)

    return df_inter


def conversion_to_block(df: pd.DataFrame, id_key: str, rule: str,
                        time_vars: Union[List[str], str], static_vars: Union[List[str], str]) -> pd.DataFrame:
    """
    Given trajectory data over multiple admissions (as specified by id), resample each admission according to time
    until the end of the admission. Resampling according to rule and apply to_time_vars.

    df: pd.DataFrame containing trajectory and static data.
    id_key: str, unique identifier per admission
    rule: str, indicates resampling rule (to be fed to pd.DataFrame.resample())
    time_vars: list of str, indicates columns of df to be resampled.
    static_vars: list of str, indicates columns of df which are static, and therefore not resampled.

    return: Dataframe with resampled vital sign data.
    """
    if "time_to_end" not in df.columns:
        raise ValueError("'time_to_end' not found in columns of dataframe. Run 'compute_time_to_end' function first.")
    assert df[id_key].is_monotonic and df.groupby(id_key).apply(
        lambda x: x["time_to_end"].is_monotonic_decreasing).all()

    # Resample admission according to time_to_end
    output = df.groupby(id_key).progress_apply(lambda x: _resample_adm(x, rule, "time_to_end", time_vars, static_vars))

    return output.reset_index(drop=True)


def convert_to_timedelta(df: pd.DataFrame, *args) -> pd.DataFrame:
    """Convert all given cols of dataframe to timedelta."""
    output = df.copy()
    for arg in args:
        output[arg] = pd.to_timedelta(df.loc[:, arg])

    return output


def _check_all_tables_exist(folder_path: str):
    """TO MOVE TO TEST"""
    try:
        assert os.path.exists(folder_path)
    except Exception:
        raise ValueError("Folder path does not exist - Input {}".format(folder_path))


def select_death_icu_acute(df, admissions_df, timedt):
    """
    Identify outcomes based on severity within the consequent 12 hours:
    a) Death
    b) Entry to ICU Careunit
    c) Transfer to hospital ward
    d) Discharge

    Params:
    - df - transfers dataframe corresponding to a particular admission.
    - timedt - datetime timedelta indicating range window of prediction

    Returns categorical encoding of the corresponding admission.
    Else returns 0,0,0,0 if a mistake is found.
    """
    # Check admission contains only one such row
    assert admissions_df.hadm_id.eq(df.name).sum() <= 1

    # Identify Last observed vitals for corresponding admission
    hadm_information = admissions_df.query("hadm_id==@df.name").iloc[0, :]
    max_vitals_obvs = hadm_information.loc["outtime"]

    # First check if death exists
    hadm_information = admissions_df.query("hadm_id==@df.name")
    if not hadm_information.empty and not hadm_information.dod.isna().all():
        time_of_death = hadm_information.dod.norm_min()
        time_from_vitals = (time_of_death - max_vitals_obvs)

        # try:
        #     assert time_from_vitals >= dt.timedelta(seconds=0)
        #
        # except AssertionError:
        #     return pd.Series(data=[0, 0, 0, 0, time_of_death], index=["De", "I", "W", "Di", "time"])

        # Check death within time window
        if time_from_vitals < timedt:
            return pd.Series(data=[1, 0, 0, 0, time_of_death], index=["De", "I", "W", "Di", "time"])

    # Otherwise, consider other transfers
    transfers_within_window = df[df["intime"].between(max_vitals_obvs, max_vitals_obvs + timedt)]

    # Consider icu transfers within window
    icu_cond1 = transfers_within_window.careunit.str.contains("(?i)ICU", na=False)  # regex ignore lowercase
    icu_cond2 = transfers_within_window.careunit.str.contains("(?i)Neuro Stepdown", na=False)
    has_icus = (icu_cond1 | icu_cond2)

    if has_icus.sum() > 0:
        icu_transfers = transfers_within_window[has_icus]
        return pd.Series(data=[0, 1, 0, 0, icu_transfers.intime.norm_min()],
                         index=["De", "I", "W", "Di", "time"])

    # Check to see if discharge has taken
    discharges = transfers_within_window.eventtype.str.contains("discharge", na=False)
    if discharges.sum() > 0:
        return pd.Series(data=[0, 0, 0, 1, transfers_within_window[discharges].intime.norm_min()],
                         index=["De", "I", "W", "Di", "time"]
                         )
    else:
        return pd.Series(data=[0, 0, 1, 0, transfers_within_window.intime.norm_min()],
                         index=["De", "I", "W", "Di", "time"]
                         )
