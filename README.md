This is a GitHub Repo for the paper "Learning of Cluster-based Feature Importance for Electronic Health Record Time-series", accepted at ICML 2022, in Baltimore, MD, US.

Our work tackled the challenging problem of identifying and characterizing disease phenotypes in multi-modal, multi-dimensional, and unevenly-sampled temporal EHR data. We proposed a novel deep learning method to cluster EHR trajectories that leverages event information to improve phenotyping. We introduced loss functions to address class imbalance and cluster collapse. Furthermore, we designed and validated a novel mechanism capable of identifying cluster-specific phenotypic importance for inputs across time and feature dimensions (i.e., which and when features contribute to events).

Link to Presentation: https://lnkd.in/dAGWWH8J
Paper: https://lnkd.in/d3kT-RRe

Work done with Tingting Zhu, Mauro Santos and Peter Watkinson at the University of Oxford. Please get in touch if you are interested and would like to discuss further or talk more generally about phenotype identification and clustering!


The Repo is structured as follows:
- all scripts are saved under "src/"
- data is assumed to be under folder "data/{DATA_NAME}"
- paths to save results and visualisations are "visualisations/{DATA_NAME}/" and "results/{DATA_NAME}"

- "src/data_processing/" details scripts for processing HAVEN (proprietary dataset) and MIMIC-IV dataset.
- "src/results/" contains main.py script that determines what results to save/how to save/...
- Similarly, "src/visualisation/" contains main.py script that determines how to print, what to print, ...
- "src/models/" contains all models considered for analysis (inc. CAMELOT and benchmarks). Each model contains a model wrapper class that has a "train" and "analyse" methods.
- Training can be done in "src/training/run_model.py" using the command "python -m src.training.run_model" using this folder as the working directory. Configuration files for data, model and results need to be edited for new experiments. The script runs for all configuration possibilities (check the individual folders for the precise configuration names". 
