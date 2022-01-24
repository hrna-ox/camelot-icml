# INSERT FILE DESCRIPTION


# ------------------------- Imports --------------------------------
import os
import datetime as dt

import pandas as pd

import src as utils
import src as test

# ------------------------ Checking Data Loaded -------------------------------
from src import SAVE_FD

try:
    assert os.path.exists(SAVE_FD + "admissions_intermediate.csv")
    assert os.path.exists(SAVE_FD + "vitals_intermediate.csv")
except Exception:
    raise ValueError(f"Run admissions_processing.py and vitals_processing.py prior to running '{__file__}'")






# ------------------------ Configuration params --------------------------------
"Global and Argument variables for Vital sign processing"
from src import DATA_FD, SAVE_FD, resampling_rule
TIME_VARS = ["intime", "outtime", "next_intime", "next_outtime", "dod"]
VITALS_TIME_VARS = ["intime", "outtime", "time_to_end_min", "time_to_end_max",
                    "time_to_end", f"sampled_time_to_end({resampling_rule})"]
# parser = argparse.ArgumentParser()
# parser.parse_args()






# ------------------------ Data Loading ------------------------------
admissions = pd.read_csv(SAVE_FD + "admissions_intermediate.csv", index_col = 0, header = 0, parse_dates = TIME_VARS)
vitals = pd.read_csv(SAVE_FD + "vitals_intermediate.csv", index_col = 0, header = 0, parse_dates = VITALS_TIME_VARS)
transfers_core = pd.read_csv(DATA_FD + "core/transfers.csv", index_col = None, header = 0,
                             parse_dates = ["intime", "outtime"])
vitals = utils.convert_to_timedelta(vitals, f"sampled_time_to_end({resampling_rule})", "time_to_end", "time_to_end_min",
                                    "time_to_end_max")

# Check correct computation of admissions
test.admissions_processed_correctly(admissions)
test.vitals_processed_correctly(vitals)




# ------------------------ Targets Processing -----------------------------
# ## transfers_subset = utils.subsetted_by(transfers_core, admissions, ["hadm_id"])


# Save vitals
vitals = vitals_subset
cohort_v3 = vitals.groupby("stay_id", as_index=False).apply(
    lambda x: x[["subject_id", "stay_id", "intime", "outtime", "chartmin", "chartmax"]].iloc[0, :]).reset_index(drop = True)
cohort_v3["hadm_id"] = cohort_v2.copy().set_index("subject_id", drop = True).loc[cohort_v3.subject_id.values, "hadm_id"].astype(int).values
vitals["hadm_id"] = cohort_v3.copy().set_index("subject_id", drop = True).loc[vitals.subject_id.values, "hadm_id"].astype(int).values

"""
Potential Outcomes will be regular admission to acute ward, ICU or death
"""
transfer_cohort = transfer_all[transfer_all.hadm_id.isin(cohort_v3.hadm_id) & transfer_all.subject_id.isin(cohort_v3.subject_id)]
hours = 12
def select_death_icu_acute(df, vitals_df, admissions_df, hours = 12):
    """
    Identify outcomed based on severity within the consequent 12 hours:
    a) Death
    b) Entry to ICU Careunit
    c) Transfer to hospital ward
    d) Discharge

    Returns categorical encoding of the corresponding admission.
    Else returns 0,0,0,0 if a mistake is found.
    """

    # Identify Last observed vitals for corresponding admission
    hadm_match = vitals_df["hadm_id"] == df.name
    subject_match = vitals_df["subject_id"] == df.subject_id.iloc[0]
    max_vitals_obvs = vitals_df[hadm_match & subject_match].chartmax.max()

    # Consider only transfers after vitals
    transfers_within_hours = df[df["intime"].between(max_vitals_obvs, max_vitals_obvs + dt.timedelta(hours = hours))]
    assert admissions_df.hadm_id.eq(df.name).sum() <= 1


    # First check if death exists
    hadm_information = admissions_df.query("hadm_id==@df.name")
    if not hadm_information.empty and not hadm_information.deathtime.isna().all():
        time_of_death = hadm_information.deathtime.min()
        time_from_vitals = (time_of_death - max_vitals_obvs).total_seconds()

        try:
            assert time_from_vitals >= 0
        except Exception:
            return pd.Series(data = [0,0,0,0], index = ["De", "I", "W", "Di"])

        if  time_from_vitals < (hours * 3600):
            return pd.Series(data = [1,0,0,0],
                                index = ["De", "I", "W", "Di"])

    # Then consider ICU
    icu_cond1 = transfers_within_hours.careunit.str.contains("(?i)ICU", na = False) #regex ignore lowercase
    # icu_cond2 = transfers_within_hours.careunit.str.contains("(?i)Neuro Stepdown", na = False) #regex ignore lowercase
    # icu_cond3 = transfers_within_hours.careunit.str.contains("(?i)Coronary Care Unit (CCU)", na = False) #regex ignore lowercase
    # icu_cond4 = transfers_within_hours.careunit.str.contains("(?i)Neuro Intermediate", na = False) #regex ignore lowercase
    has_icus = (icu_cond1).sum() > 0

    if has_icus:
        return pd.Series(data=[0, 1, 0, 0],
                            index =["De", "I", "W", "Di"])

    # Check to see if discharge has taken
    is_discharged = transfers_within_hours.eventtype.str.contains("discharge", na = False).sum() > 0
    if is_discharged:
        return pd.Series(data = [0, 0, 0, 1],
                            index=["De", "I", "W", "Di"]
                            )
    else:
        return pd.Series(data = [0, 0, 1, 0],
                            index=["De", "I", "W", "Di"]
                            )

# Need to include Death
outcomes_try_1 = transfer_cohort.groupby("hadm_id", as_index = True).apply(
    lambda x: select_death_icu_acute(x, vitals, admissions_all, hours))

# Ignore those patients with mistakes
outcomes = outcomes_try_1[~ outcomes_try_1.eq(0).all(axis = 1)]
vitals_df = vitals[vitals["hadm_id"].isin(outcomes.index.values)]
cohort = cohort_v3[cohort_v3["hadm_id"].isin(outcomes.index.values)]

assert set(vitals_df.hadm_id.values) == (set(outcomes.index))
assert set(cohort.hadm_id.values) == (set(outcomes.index))

# Save to output variables
os.chdir("mimic-iv-ed/data/")

if not os.path.exists("derived/"):
    os.makedirs("derived/")
vitals_df.to_csv("derived/vitals.csv", index = True, header = True)
outcomes.to_csv("derived/outcomes.csv", index = True, header = True)
cohort.to_csv("derived/cohort.csv", index = True, header = True)
