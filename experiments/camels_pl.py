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
    path_entities = "../../HYMC/data/basin_id/basins_camels_pl_354.txt"
    path_data = "../../HYMC/data/CAMELS_PL"
    path_additional_features = "../../HYMC/data/CAMELS_PL/pet_hargreaves.pickle"
    input_variables = ["precipitation_mean", "pet(mm/day)", "temperature_mean"]
    target_variables = ["discharge_spec_obs"]
    # Periods start 1 year earlier to account for warmup_period=365 days.
    # Actual NSE evaluation starts after warmup:
    #   training NSE: 2005-11-01 – 2015-12-31  (matches LSTM training period)
    #   testing  NSE: 2016-01-01 – 2024-10-31  (matches LSTM testing period)
    training_period = ["2004-11-01", "2015-12-31"]
    testing_period  = ["2015-01-01", "2024-10-31"]
    batch_size = 20
    warmup_period = 365

    # 5 random seeds to account for parameter uncertainty (ensemble approach)
    random_seeds = [42, 110, 123, 456, 789]

    model = "HBV"          # Bucket, HBV, Nonsense, SHM
    dataset = "CAMELS_PL"  # CAMELS_GB, CAMELS_US, CAMELS_DE
    optimizer = "DREAM"    # DREAM, SCE
    loss_function = "gaussian_likelihood_meas_error_out"

    # -----------------------------------------------------------------------------------------------------------------
    # Get the classes associated with the running configuration
    basin_dataset = get_dataset(dataset)
    obj_func, maximize = get_loss_function(loss_function)
    hydrological_model = get_model(model)
    optimization_algorithm = get_optimizer(optimizer)

    # Read basin IDs
    basins_id = read_basins_id(path_entities=path_entities)

    # Process time periods in case they are given as custom ranges
    if isinstance(training_period, str):
        training_period = basin_dataset.create_custom_periods(training_period)
    if isinstance(testing_period, str):
        testing_period = basin_dataset.create_custom_periods(testing_period)

    # Filter basins that do not have target data during training or testing periods.
    selected_basins_id = basin_dataset.check_basins(
        path_data=path_data,
        basins_id=basins_id,
        target_variables=target_variables,
        training_period=training_period,
        testing_period=testing_period,
        warmup_period=warmup_period,
    )

    batches = [selected_basins_id[i : i + batch_size] for i in range(0, len(selected_basins_id), batch_size)]

    # =================================================================================================================
    # Loop over seeds
    # =================================================================================================================
    total_start = time.time()

    for seed_idx, random_seed in enumerate(random_seeds):
        print(f"\n{'='*70}")
        print(f"  SEED {random_seed}  ({seed_idx + 1}/{len(random_seeds)})")
        print(f"{'='*70}")

        path_output = f"../results/HBV_CAMELS_PL_seed_{random_seed}/"
        create_folder(folder_path=path_output)
        set_random_seed(seed=random_seed)

        dfs = []
        seed_start = time.time()

        for batch_idx, basin_batch in enumerate(batches):
            training_object = {}
            testing_object = {}

            for basin in basin_batch:
                training_object[basin] = basin_dataset(
                    model=hydrological_model(),
                    path_data=path_data,
                    basin_id=basin,
                    input_variables=input_variables,
                    target_variables=target_variables,
                    time_period=training_period,
                    obj_func=obj_func,
                    warmup_period=warmup_period,
                    path_additional_features=path_additional_features,
                )

                testing_object[basin] = basin_dataset(
                    model=hydrological_model(),
                    path_data=path_data,
                    basin_id=basin,
                    input_variables=input_variables,
                    target_variables=target_variables,
                    time_period=testing_period,
                    obj_func=obj_func,
                    warmup_period=warmup_period,
                    path_additional_features=path_additional_features,
                )

            # Run calibration in parallel -----------------------------------------------------------------------
            calibrate_multiple_basins(
                training_object=training_object,
                optimization_method=optimization_algorithm,
                path_output=path_output,
                random_state=random_seed,
            )

            # Summarize results ---------------------------------------------------------------------------------
            hyd_model = hydrological_model()
            df_calibration = pd.DataFrame(
                index=range(len(basin_batch)),
                columns=["basin_id", "NSE_training"] + list(hyd_model.parameter_ranges) + ["NSE_testing"],
            )

            for i, basin in enumerate(basin_batch):
                # Extract best calibrated parameters
                file_name = path_output + model + "_" + optimizer + "_" + str(basin)
                results = spotpy.analyser.load_csv_results(file_name)
                calibrated_param = spotpy.analyser.get_best_parameterset(results, maximize=maximize)[0]

                # Training period ----------------------------------
                q_sim = training_object[basin].simulation(calibrated_param)
                q_obs = training_object[basin].evaluation()
                evaluation = q_obs[warmup_period:][training_object[basin].data_split[warmup_period:]]
                simulation = q_sim[warmup_period:][training_object[basin].data_split[warmup_period:]]
                mask_nans = ~np.isnan(evaluation)
                NSE_training = nse_loss(
                    evaluation=evaluation[mask_nans].flatten(),
                    simulation=simulation[mask_nans].flatten(),
                )

                # Testing period -----------------------------------
                q_sim = testing_object[basin].simulation(calibrated_param)
                q_obs = testing_object[basin].evaluation()
                evaluation = q_obs[warmup_period:][testing_object[basin].data_split[warmup_period:]]
                simulation = q_sim[warmup_period:][testing_object[basin].data_split[warmup_period:]]
                mask_nans = ~np.isnan(evaluation)
                NSE_testing = nse_loss(
                    evaluation=evaluation[mask_nans].flatten(),
                    simulation=simulation[mask_nans].flatten(),
                )

                df_calibration.loc[i] = [basin, NSE_training] + list(calibrated_param) + [NSE_testing]

            dfs.append(df_calibration)
            print(f"  Batch {batch_idx + 1}/{len(batches)} done")

        # Save summary for this seed
        combined_df = pd.concat(dfs)
        combined_df.to_csv(path_output + model + "_" + optimizer + "_summary.csv", index=False)

        seed_elapsed = time.time() - seed_start
        print(f"  Seed {random_seed} done in {int(seed_elapsed)}s → {path_output}")

    total_elapsed = time.time() - total_start
    print(f"\nAll {len(random_seeds)} seeds completed in {int(total_elapsed)}s ({total_elapsed/60:.1f} min)")
    print("Results saved to:")
    for s in random_seeds:
        print(f"  ../results/HBV_CAMELS_PL_seed_{s}/")
