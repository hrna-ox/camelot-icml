# Processing script for initial ED admission processing.

import os
import pandas as pd

import src.data_processing.MIMIC.data_utils as utils

####################################################
"""

Processing Steps:

1. Compute recorded intime and outimes for each ED admission.
2. Select admissions with ED as the first admission.
3. Remove admissions admitted to special wards, including Partum and Psychiatry. Compute next transfer information.
4. Add patient core information.
5. Remove admissions without triage information.


Other notes.
ROW SUBSETTING COULD BE IMPROVED SOMEHOW
"""

# ------------------------------------ // --------------------------------------
"""
List of variables used for processing which should be fixed.

Data_FD: where the data is saved.
SAVE_FD: folder path of interim data saving.
ID_COLUMNS: identifiers for admissions, patients and hospital stays.
TIME_COLUMNS: list of datetime object columns.
WARDS_TO_REMOVE: list of special wards where patients were transferred to and which represent unique populations. This
list includes Partum and Psychiatry wards, as well as further ED observations, which generally take place when 
the hospital is full.
AGE_LOWERBOUND: minimum age of patients.
PATIENT_INFO: characteristic information for each patient.
NEXT_TRANSFER_INFO: list of important info to keep related to the subsequent transfer from ED.
"""
DATA_FD = "data/MIMIC/"
SAVE_FD = DATA_FD + "interim/"
ID_COLUMNS = ["subject_id", "hadm_id", "stay_id"]
TIME_COLUMNS = ["intime", "outtime", "charttime", "deathtime"]
WARDS_TO_REMOVE = ["Unknown", "Emergency Department", "Obstetrics Postpartum",
                   "Obstetrics Antepartum", "Obstetrics (Postpartum & Antepartum)",
                   "Psychiatry", "Labor & Delivery", "Observation", "Emergency Department Observation"]
AGE_LOWERBOUND = 18
PATIENT_INFO = ["gender", "anchor_age", "anchor_year", "dod"]
NEXT_TRANSFER_INFO = ["transfer_id", "eventtype", "careunit", "intime", "outtime"]

if not os.path.exists(SAVE_FD):
    os.makedirs(SAVE_FD)

# ------------------------------------- // -------------------------------------
if __name__ == "__main__":

    """
    First, Tables are Loaded. We load 4 tables:
    
    - patients_core: from core/patients filepath. This is a dataframe of patient centralised admission information. 
    Cohort information for each patient is computed, as well as a unique id which is consistent across all other tables.
    
    - transfer_core: from core/transfers.csv filepath. This is a dataframe with a list of transfers for each patient.
    Includes admissions to ED, but also transfers to wards in the hospital, ICUs, etc...
    
    - admissions_ed: from ed/edstays.csv filepath. This is a dataframe of patient information indicating relevant
    information for any ED admission.
    
    - triage_ed: from ed/triage.csv filepath. This is a dataframe of patient ED admission indicating triage assessments.
    """

    # Hospital Core
    patients_core = pd.read_csv(DATA_FD + "core/patients.csv", index_col=None, header=0, low_memory=False)
    transfers_core = pd.read_csv(DATA_FD + "core/transfers.csv", index_col=None, header=0, low_memory=False,
                                 parse_dates=["intime", "outtime"])

    # ED Admission
    admissions_ed = pd.read_csv(DATA_FD + "ed/edstays.csv", index_col=None, header=0, low_memory=False,
                                parse_dates=["intime", "outtime"])
    triage_ed = pd.read_csv(DATA_FD + "ed/triage.csv", index_col=None, header=0, low_memory=False)

    # ------------------------------------- // -------------------------------------
    """
    Process Admission data according to multiple steps.
    
    Step 1: Remove double admission counts. Select the latest intime. If there are multiple such intimes, 
    select the last outtime.
    Consider only these admissions.
    """

    # Compute recorded admission intimes and outtimes. Respectively, select latest intime and outtime.
    admissions_intime_ed = utils.endpoint_target_ids(admissions_ed, "subject_id", "intime")
    admissions_outtime_ed = utils.endpoint_target_ids(admissions_intime_ed, "subject_id", "outtime")

    admissions_ed_S1 = utils.subsetted_by(admissions_ed, admissions_outtime_ed,
                                          ["stay_id"])  # last admission information
    admissions_ed_S1.to_csv(SAVE_FD + "admissions_S1.csv", index=True, header=True)

    """
    Identify those admissions where patients were directly sent to Emergency Department, i.e., the first intime
    is in the Emergency Department. 
    Subset to these admissions.
    """
    # Identify first wards for all admissions to hospital
    transfers_first_ward = utils.endpoint_target_ids(transfers_core, "subject_id", "intime", mode="min")
    ed_first_transfer = transfers_first_ward[(transfers_first_ward["eventtype"] == "ED") &
                                             (transfers_first_ward["careunit"] == "Emergency Department")]

    # Subset to admissions with ED as first transfer
    admissions_ed_S2 = utils.subsetted_by(admissions_ed_S1, ed_first_transfer,
                                          ["subject_id", "hadm_id", "intime", "outtime"])
    transfers_ed_S2 = utils.subsetted_by(transfers_core, admissions_ed_S2, ["subject_id", "hadm_id"])
    admissions_ed_S2.to_csv(SAVE_FD + "admissions_S2.csv", index=True, header=True)

    """
    Consider only those admissions for which they did not have a subsequent transfer to a Special ward, which includes
    Partum and Psychiatry wards. The full list of wards is identified in WARDS TO REMOVE
    """
    # Remove admissions transferred to irrelevant wards (Partum, Psychiatry). Furthermore, EDObs is also special.
    # Missing check that second intime is after ED outtime
    transfers_second_ward = utils.compute_second_transfer(transfers_ed_S2, "subject_id", "intime",
                                                          transfers_ed_S2.columns)
    transfers_to_relevant_wards = transfers_second_ward[~ transfers_second_ward.careunit.isin(WARDS_TO_REMOVE)]
    admissions_ed_S3 = utils.subsetted_by(admissions_ed_S2, transfers_to_relevant_wards, ["subject_id", "hadm_id"])

    # ADD patient core information and next Transfer Information.
    patients_S3 = admissions_ed_S3.subject_id.values
    admissions_ed_S3.loc[:, PATIENT_INFO] = patients_core.set_index("subject_id").loc[patients_S3, PATIENT_INFO].values

    for col in NEXT_TRANSFER_INFO:
        admissions_ed_S3.loc[:, "next_" + col] = transfers_to_relevant_wards.set_index("subject_id").loc[
            patients_S3, col].values

    # Compute age and save
    admissions_ed_S3["age"] = admissions_ed_S3.intime.dt.year - admissions_ed_S3["anchor_year"] + admissions_ed_S3[
        "anchor_age"]
    admissions_ed_S3.to_csv(SAVE_FD + "admissions_S3.csv", index=True, header=True)

    """
    Step 4: Patients must have an age older than AGE LOWERBOUND
    """
    # Compute age and Remove below AGE LOWERBOUND
    admissions_ed_S4 = admissions_ed_S3[admissions_ed_S3["age"] >= AGE_LOWERBOUND]
    admissions_ed_S4.to_csv(SAVE_FD + "admissions_S4.csv", index=True, header=True)

    """
    Step 5: Add ESI information, and subset to patients with ESI values and between 2, 3, 4.
    ESI values of 1 and 5 are edge cases (nothing wrong with them, or in extremely critical condition).
    """
    # Compute and remove ESI NAN, ESI 1 and ESI 5 and save
    admissions_ed_S4["ESI"] = triage_ed.set_index("stay_id").loc[admissions_ed_S4.stay_id.values, "acuity"].values
    admissions_ed_S5 = admissions_ed_S4[~ admissions_ed_S4["ESI"].isna()]
    admissions_ed_S5 = admissions_ed_S5[~ admissions_ed_S5["ESI"].isin([1, 5])]

    # Save data
    admissions_ed_S5.to_csv(SAVE_FD + "admissions_intermediate.csv", index=True, header=True)
