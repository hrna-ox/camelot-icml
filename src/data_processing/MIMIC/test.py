#!/usr/bin/env python3
"""
Test file to check data has been correctly processed
"""
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

def test_entrance_before_exit(entrance: pd.Series, exit: pd.Series) -> bool:
    """Check entrance times are NOT observed after Exit times, or both values are missing"""

    diff = (exit - entrance).dt.total_seconds()  # Difference in seconds
    missing = exit.isna() & entrance.isna()  # bool indicating both values are missing

    return (diff.ge(0) | missing).all()


def test_exit_before_next_entrance(cur_exit: pd.Series, next_entrance: pd.Series) -> bool:
    """Check exit of current admission is observed before entrance to consequent admission, or next is missing"""

    diff = (next_entrance - cur_exit).dt.total_seconds()  # Difference in seconds
    missing = next_entrance.isna()   # bool indicating next admission is missing

    return (diff.ge(0) | missing).all()


def test_time_before_death(exit: pd.Series, death: pd.Series) -> bool:
    """Check admission occurs before exit, or one of the values is missing"""

    diff = (death - exit.astype("datetime64[D]")).dt.total_seconds()  # Difference in seconds
    missing = exit.isna() | death.isna()  # bool indicating both values are missing

    return (diff.ge(0) | missing).all()


def test_is_unique_ids(df: pd.DataFrame, *args) -> bool:
    """Check whether there are any duplicate values across all id columns"""
    output = True

    for arg in args:  # Iterate through each column
        has_repeated = df[arg].dropna().duplicated().sum() > 0
        if has_repeated:
            print(f"There are duplicate values for id {arg}")

        output = output and not has_repeated

    return output


def test_is_complete_ids(df: pd.DataFrame, *args) -> bool:
    """Check no missing values across id columns"""
    output = True

    for arg in args:  # Iterate through each column
        has_missing = df[arg].isna().sum() > 0
        if has_missing:
            print(f"There are missing values values for id {arg}")

        output = output and not has_missing

    return output


def admissions_processed_correctly(df: pd.DataFrame):
    """
    Function to check intermediate processing of admissions is correct. The following are done:
    1. Entrance times for ED and subsequent admissions occur prior to exit.
    2. Entrance Time for consequent admissions (when exist) do not take place after ED Exit time.
    2. Identifiers are unique.
    3. Subject and Stay id and ED times are complete.
    """

    # Within admission time check
    assert test_entrance_before_exit(df["intime"], df["outtime"])
    assert test_entrance_before_exit(df["next_intime"], df["next_outtime"])

    # Between consequent admission time check
    assert test_exit_before_next_entrance(df["outtime"], df["next_intime"])

    # Check Death observed any admission times
    assert test_time_before_death(df["outtime"], df["dod"])
    assert test_time_before_death(df["next_intime"], df["dod"])

    # Uniqueness of main id columns
    assert test_is_unique_ids(df, "subject_id", "hadm_id", "stay_id", "next_transfer_id")

    # Completeness of id columns
    assert test_is_complete_ids(df, "subject_id", "stay_id", "intime", "outtime")

    print("Admissions correctly computed! Safe to go ahead.")


def vitals_processed_correctly(df: pd.DataFrame):
    """
    Function to check intermediate processing of vitals is correct. The following are done:
    1. Intime before Outtime
    2. Time to End Max/Min fall within intime/outtime ED time.
    3. Sampled Time to End falls within intime/outtime ED time.
    4. Identifiers are complete.
    """

    # Intime before Outtime
    assert test_entrance_before_exit(df["intime"], df["outtime"])

    # Time to End within intime/outtime
    assert test_entrance_before_exit(df["outtime"] - df["time_to_end_min"], df["outtime"])
    assert test_time_before_death(df["intime"], df["outtime"] - df["time_to_end_max"])

    # Similar to sampled time to end
    resampling_rule = "1H"
    col = f"sampled_time_to_end({resampling_rule})"
    assert test_entrance_before_exit(df["outtime"] - df[col], df["outtime"])
    assert test_time_before_death(df["intime"], df["outtime"] - df[col])

    # Completeness of id columns
    assert test_is_complete_ids(df, "stay_id", col, "intime", "outtime", "time_to_end_min", "time_to_end_max")

    print("Vitals correctly computed! Safe to go ahead.")