# Import necessary packages
import os
import sys
import time

import numpy as np
import pandas as pd
import spotpy

os.chdir(sys.path[0])
sys.path.append("..")

from hymc.aux_functions.evaluation_metrics import nse_loss
from hymc.aux_functions.modelcalibration import calibrate_multiple_basins
from hymc.aux_functions.utils import create_folder, read_basins_id, set_random_seed
from hymc.datasetzoo import get_dataset
from hymc.modelzoo import get_model
from hymc.optimizationmethods import get_loss_function, get_optimizer

# -----------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize information
    path_entities = "../../Hy2DL/data/basin_id/basins_camels_gb_5.txt"
    path_data = "../../Hy2DL/data/CAMELS_GB"
    input_variables = ["precipitation", "peti", "temperature"]
    target_variables = ["discharge_spec"]
    training_period = ["1980-10-01", "1997-12-31"]
    testing_period = ["1997-01-01", "2008-12-31"]
    batch_size = 20
    warmup_period = 365
    random_seed = 42

    model = "SHM" # Bucket, HBV, Nonsense, SHM
    dataset = "CAMELS_GB" # CAMELS_GB, CAMELS_US, CAMELS_DE
    optimizer = "DREAM" # DREAM, SCE
    loss_function = "gaussian_likelihood_meas_error_out" # rmse, gaussian_likelihood_meas_error_out

    # Path to save results
    path_output = "../results/conceptual_models_CAMELS_GB/"

    # -----------------------------------------------------------------------------------------------------------------
    set_random_seed(seed=random_seed)
    create_folder(folder_path= path_output)

    # Get the classes associated with the running configuration
    basin_dataset = get_dataset(dataset)
    obj_func, maximize = get_loss_function(loss_function)
    hydrological_model = get_model(model)
    optimization_algorithm = get_optimizer(optimizer)

    # Read information 
    basins_id = read_basins_id(path_entities = path_entities)

    # Process time periods in case they are given as custom ranges
    if isinstance(training_period, str):
        training_period = basin_dataset.create_custom_periods(training_period)
    if isinstance(testing_period, str):
        testing_period = basin_dataset.create_custom_periods(testing_period)

    # Filter basins that do not have target data during training or testing periods.
    selected_basins_id = basin_dataset.check_basins(path_data=path_data,
                                                    basins_id=basins_id,
                                                    target_variables=target_variables,
                                                    training_period=training_period,
                                                    testing_period=testing_period,
                                                    warmup_period=warmup_period
                                                    )

    # Process the basins in batches (avoid memory issues)
    dfs = []
    batches = [selected_basins_id[i : i + batch_size] for i in range(0, len(selected_basins_id), batch_size)]

    start_time = time.time()
    for basin_batch in batches:
        training_object = {}
        testing_object = {}

        for basin in basin_batch:
            training_object[basin] = basin_dataset(model=hydrological_model(),
                                                   path_data=path_data,
                                                   basin_id=basin,
                                                   input_variables=input_variables,
                                                   target_variables=target_variables,
                                                   time_period=training_period,
                                                   obj_func=obj_func,
                                                   warmup_period=warmup_period
                                                   )

            testing_object[basin] = basin_dataset(model=hydrological_model(),
                                                  path_data=path_data,
                                                  basin_id=basin,
                                                  input_variables=input_variables,
                                                  target_variables=target_variables,
                                                  time_period=testing_period,
                                                  obj_func=obj_func,
                                                  warmup_period=warmup_period
                                                  )

        # Run the calibration of the different basins in parallel --------------------------------------------------
        calibrate_multiple_basins(training_object=training_object,
                                  optimization_method=optimization_algorithm,
                                  path_output=path_output,
                                  random_state=random_seed
                                  )

        # Process and summarize the results -------------------------------------------------------------------------
        hyd_model = hydrological_model()
        df_calibration = pd.DataFrame(
            index=range(len(basin_batch)),
            columns=["basin_id", "NSE_training"] + list(hyd_model.parameter_ranges) + ["NSE_testing"]
            )

        for i, basin in enumerate(basin_batch):
            # extract calibrated parameters
            file_name = path_output + model + "_" + optimizer + "_" + str(basin)
            results = spotpy.analyser.load_csv_results(file_name)
            calibrated_param = spotpy.analyser.get_best_parameterset(results, maximize=maximize)[0]

            # Training period ------------------------------------------
            q_sim = training_object[basin].simulation(calibrated_param)
            q_obs = training_object[basin].evaluation()

            # Calculate loss
            evaluation = q_obs[warmup_period:][training_object[basin].data_split[warmup_period:]]
            simulation = q_sim[warmup_period:][training_object[basin].data_split[warmup_period:]]
            mask_nans = ~np.isnan(evaluation)
            NSE_training = nse_loss(
                evaluation=evaluation[mask_nans].flatten(), simulation=simulation[mask_nans].flatten()
            )

            # Testing period ------------------------------------------
            q_sim = testing_object[basin].simulation(calibrated_param)
            q_obs = testing_object[basin].evaluation()

            # Calculate loss
            evaluation = q_obs[warmup_period:][testing_object[basin].data_split[warmup_period:]]
            simulation = q_sim[warmup_period:][testing_object[basin].data_split[warmup_period:]]
            mask_nans = ~np.isnan(evaluation)
            NSE_testing = nse_loss(
                evaluation=evaluation[mask_nans].flatten(), simulation=simulation[mask_nans].flatten()
            )

            # Save the result of the basin
            df_calibration.loc[i] = [basin, NSE_training] + list(calibrated_param) + [NSE_testing]

        # Dataframe of the batch
        dfs.append(df_calibration)

    # Save the results
    combined_df = pd.concat(dfs)
    combined_df.to_csv(path_output + model + "_" + optimizer + "_summary.csv", index=False)
    # Calculate and print the calibration time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total calibration time: {int(elapsed_time)} seconds")
