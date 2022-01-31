# Processing


import datetime as dt
# ------------------------- Imports --------------------------------
import os

import pandas as pd

import src.data_processing.MIMIC.data_utils as utils
import src.data_processing.MIMIC.test as test

# ------------------------ Checking Data Loaded -------------------------------
from src.data_processing.MIMIC.admissions_processing import SAVE_FD, DATA_FD
from src.data_processing.MIMIC.vitals_processing import resampling_rule

try:
    assert os.path.exists(SAVE_FD + "admissions_intermediate.csv")
    assert os.path.exists(SAVE_FD + "vitals_intermediate.csv")
except Exception:
    raise ValueError(f"Run admissions_processing.py and vitals_processing.py prior to running '{__file__}'")

# ------------------------ Configuration params --------------------------------
"""
Global and Argument variables for Vital sign processing.
"""
TIME_VARS = ["intime", "outtime", "next_intime", "next_outtime", "dod"]
VITALS_TIME_VARS = ["intime", "outtime", "time_to_end_min", "time_to_end_max",
                    "time_to_end", f"sampled_time_to_end({resampling_rule})"]

# ------------------------ Data Loading ------------------------------
admissions = pd.read_csv(SAVE_FD + "admissions_intermediate.csv", index_col=0, header=0, parse_dates=TIME_VARS)
vitals = pd.read_csv(SAVE_FD + "vitals_intermediate.csv", index_col=0, header=0, parse_dates=VITALS_TIME_VARS)
transfers_core = pd.read_csv(DATA_FD + "core/transfers.csv", index_col=None, header=0,
                             parse_dates=["intime", "outtime"])
vitals = utils.convert_to_timedelta(vitals, f"sampled_time_to_end({resampling_rule})", "time_to_end", "time_to_end_min",
                                    "time_to_end_max")

# Check correct computation of admissions
test.admissions_processed_correctly(admissions)
test.vitals_processed_correctly(vitals)

# ------------------------ Targets Processing -----------------------------
admissions_subset = utils.subsetted_by(admissions, vitals, ["stay_id"])
transfers_subset = utils.subsetted_by(transfers_core, admissions_subset, ["subject_id", "hadm_id"])
vitals["chartmax"] = vitals["outtime"] - vitals["time_to_end"]
vitals["hadm_id"] = admissions.set_index("stay_id").loc[vitals.stay_id.values, "hadm_id"].values


"""
Potential Outcomes will be regular admission to acute ward, ICU or death
"""
time_window_1 = dt.timedelta(hours=4)
time_window_2 = dt.timedelta(hours=12)
time_window_3 = dt.timedelta(hours=24)
time_window_4 = dt.timedelta(hours=48)

# Need to include Death
outcomes_4_hours = transfers_subset.groupby("hadm_id", as_index=True).apply(
            lambda x: utils.select_death_icu_acute(x, admissions_subset, time_window_1))
outcomes_12_hours = transfers_subset.groupby("hadm_id", as_index=True).apply(
            lambda x: utils.select_death_icu_acute(x, admissions_subset, time_window_2))
outcomes_24_hours = transfers_subset.groupby("hadm_id", as_index=True).apply(
            lambda x: utils.select_death_icu_acute(x, admissions_subset, time_window_3))
outcomes_48_hours = transfers_subset.groupby("hadm_id", as_index=True).apply(
            lambda x: utils.select_death_icu_acute(x, admissions_subset, time_window_4))


print(outcomes_4_hours.iloc[:, :-1].sum(axis=0))
print(outcomes_12_hours.iloc[:, :-1].sum(axis=0))
print(outcomes_24_hours.iloc[:, :-1].sum(axis=0))
print(outcomes_48_hours.iloc[:, :-1].sum(axis=0))

# Ensure all patients have only one class
assert outcomes_4_hours.iloc[:, :-1].sum(axis=1).eq(1).all()
assert outcomes_12_hours.iloc[:, :-1].sum(axis=1).eq(1).all()
assert outcomes_24_hours.iloc[:, :-1].sum(axis=1).eq(1).all()
assert outcomes_48_hours.iloc[:, :-1].sum(axis=1).eq(1).all()

# Subset vitals and admission data
admissions_keep = outcomes_4_hours.index.tolist()
admissions_final = admissions_subset[admissions_subset.hadm_id.isin(admissions_keep)]
vitals_final = vitals[vitals.hadm_id.isin(admissions_keep)]

static_vars = ["gender", "age", "ESI"]
vitals_final[static_vars] = admissions_final.set_index("hadm_id").loc[vitals_final.hadm_id.values, static_vars].values
vitals_final["gender"] = vitals_final.loc[:, "gender"].replace(to_replace=["M", "F"], value=[1,0])
vitals_final["charttime"] = vitals_final.loc[:, "outtime"].values - vitals_final.loc[:, "sampled_time_to_end(1H)"].values

# Save to output variables
process_fd = DATA_FD + "processed/"

if not os.path.exists(process_fd):
    os.makedirs(process_fd)

# Save general
vitals_final.to_csv(SAVE_FD + "vitals_final.csv", index=True, header=True)
admissions_final.to_csv(SAVE_FD + "admissions_final.csv", index=True, header=True)

# Save for input
vitals_final.to_csv(process_fd + "vitals_process.csv", index=True, header=True)
admissions_final.to_csv(process_fd + "admissions_process.csv", index=True, header=True)
outcomes_4_hours.to_csv(process_fd + "outcomes_4h_process.csv", index=True, header=True)
outcomes_12_hours.to_csv(process_fd + "outcomes_12h_process.csv", index=True, header=True)
outcomes_24_hours.to_csv(process_fd + "outcomes_24h_process.csv", index=True, header=True)
outcomes_48_hours.to_csv(process_fd + "outcomes_48h_process.csv", index=True, header=True)
