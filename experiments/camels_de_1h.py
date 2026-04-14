# Import necessary packages
import os
import sys
import time
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import spotpy

from hymc.aux_functions.evaluation_metrics import nse_loss
from hymc.aux_functions.modelcalibration import calibrate_multiple_basins
from hymc.aux_functions.utils import create_folder, read_basins_id, set_random_seed
from hymc.datasetzoo import get_dataset
from hymc.modelzoo import get_model
from hymc.optimizationmethods import get_loss_function, get_optimizer

# -----------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize information
    path_entities = "../data/basin_id/basins_camels_de_1h_1611.txt"
    path_data = "../data/CAMELS-DE-1h"
    path_additional_features = "../data/CAMELS_DE_1h/pet_hargreaves.nc"
    input_variables = ["precipitation_mean_gapfilled", "pet", "air_temperature_mean"]
    target_variables = ["discharge_spec_obs"]
    training_period = ["2004-01-01 01:00:00", "2019-12-31 23:00:00"]
    testing_period = ["2019-01-01 01:00:00", "2024-12-31 23:00:00"] # the first year is warmup, so actual testing / NSE calculation starts 2000-01-01
    batch_size = 96
    warmup_period = 8760 # 1 year of hourly data

    # get random seed from environment variable (helpful to calibrate ensembles on a HPC)
    seed = os.environ.get("HYMC_SEED", 110)
    random_seed = int(seed)

    model = "HBV" # Bucket, HBV, Nonsense, SHM
    dataset = "CAMELS_DE_1h" # CAMELS_GB, CAMELS_US, CAMELS_DE, CAMELS_DE_1h
    optimizer = "DREAM" # DREAM, SCE
    loss_function = "gaussian_likelihood_meas_error_out" # rmse, gaussian_likelihood_meas_error_out

    # Path to save results
    path_output = f"../results/hbv_CAMELS_DE_1h_benchmark_seed_{random_seed}/"

    # -----------------------------------------------------------------------------------------------------------------
    set_random_seed(seed=random_seed)
    create_folder(folder_path=path_output)

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

    print(f"Checking {len(basins_id)} for calibration and testing data.")
    
    # Filter basins that do not have target data during training or testing periods.
    selected_basins_id = basin_dataset.check_basins(path_data=path_data,
                                                    basins_id=basins_id,
                                                    target_variables=target_variables,
                                                    training_period=training_period,
                                                    testing_period=testing_period,
                                                    warmup_period=warmup_period
                                                    )

    print(f"After checking, {len(selected_basins_id)} / {len(basins_id)} have calibration and testing data.")
     
    # Process the basins in batches (avoid memory issues)
    dfs = []
    batches = [selected_basins_id[i : i + batch_size] for i in range(0, len(selected_basins_id), batch_size)]

    start_time = time.time()
    for i, basin_batch in enumerate(batches):
        training_object = {}
        testing_object = {}

        for basin in basin_batch:
            training_object[basin] = basin_dataset(model=hydrological_model(resolution="hourly"),
                                                   path_data=path_data,
                                                   basin_id=basin,
                                                   input_variables=input_variables,
                                                   target_variables=target_variables,
                                                   time_period=training_period,
                                                   obj_func=obj_func,
                                                   warmup_period=warmup_period,
                                                   path_additional_features=path_additional_features
                                                   )

            testing_object[basin] = basin_dataset(model=hydrological_model(resolution="hourly"),
                                                  path_data=path_data,
                                                  basin_id=basin,
                                                  input_variables=input_variables,
                                                  target_variables=target_variables,
                                                  time_period=testing_period,
                                                  obj_func=obj_func,
                                                  warmup_period=warmup_period,
                                                  path_additional_features=path_additional_features
                                                  )

        # Run the calibration of the different basins in parallel (silently and restartable)-------------------------
        unprocessed_training = {}
        for basin in basin_batch:
            # Spotpy saves outputs with a .csv extension
            expected_file = f"{path_output}/{model}_{optimizer.lower()}_{str(basin)}.csv"
            if not os.path.exists(expected_file):
                unprocessed_training[basin] = training_object[basin]

        if not unprocessed_training:
            print(f"Batch {i+1}/{len(batches)} already processed. Skipping calibration.")
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]: Start processing batch {i+1}/{len(batches)}.")
            # Run the calibration of ONLY the unprocessed basins in parallel (silently)
            with open(os.devnull, 'w') as f, redirect_stdout(f):
                calibrate_multiple_basins(training_object=unprocessed_training,
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
            file_name = path_output + model + "_" + optimizer.lower() + "_" + str(basin)
            results = spotpy.analyser.load_csv_results(file_name)
            
            with open(os.devnull, 'w') as f, redirect_stdout(f):
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
