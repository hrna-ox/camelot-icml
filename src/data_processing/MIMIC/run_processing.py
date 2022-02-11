"""
Process MIMIC-ED Data file.

- Admissions are processed first.
- Vital observations are consequently processed based on existing observations.
- Finally, outcome targets are processed.
"""
import os
import src.data_processing.MIMIC.admissions_processing as admissions_processing
import src.data_processing.MIMIC.vitals_processing as vitals_processing
import src.data_processing.MIMIC.outcomes_processing as outcomes_processing

print("Current Directory: ", os.getcwd())

def main():


    # Run admissions if not processed
    if not os.path.exists("data/MIMIC/interim/admissions_intermediate.csv"):
    	admissions_processing.main()

    # Run vitals
    if not os.path.exists("data/MIMIC/interim/vitals_intermediate.csv"):
    	vitals_processing.main()

    # Run outcomes
    if not os.path.exists("data/MIMIC/interim/vitals_final.csv"):
    	outcomes_processing.main()

    pass


# Run processing for admission, vitals and outcomes.
if __name__ == "__main__":
    main()
