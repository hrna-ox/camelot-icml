"""
Process MIMIC-ED Data file.

- Admissions are processed first.
- Vital observations are consequently processed based on existing observations.
- Finally, outcome targets are processed.
"""

import src.data_processing.MIMIC.admissions_processing as admissions_processing
import src.data_processing.MIMIC.vitals_processing as vitals_processing
import src.data_processing.MIMIC.outcomes_processing as outcomes_processing


def main():
    # Run admissions
    admissions_processing.main()

    # Run vitals
    vitals_processing.main()

    # Run outcomes
    outcomes_processing.main()

    pass


# Run processing for admission, vitals and outcomes.
if __name__ == "__main__":
    main()
