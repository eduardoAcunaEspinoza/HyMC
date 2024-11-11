import multiprocessing
from typing import Dict


def calibrate_multiple_basins(training_object: Dict, optimization_method, path_output: str, **kwargs):
    """Run the calibration of the different basins. 
    
    The calibration is done in parallel, assigning one core per basin to speed up the code.
    
    Parameters
    ----------
    training_object: dict
        Dictionary indexed by the basin id, with the training object for each basin
    optimization_method: class
        Class with the optimization method
    path_output: str
        Path where the calibration files will be stored
    kwargs: optional
        Additional keyword arguments for the optimizer
    
    """
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    pool.starmap(
        calibrate_single_basin,
        [(basin, optimization_method, path_output, kwargs)  for basin in training_object.values()],
    )

    # Close the pool after processing
    pool.close()
    pool.join()


def calibrate_single_basin(calibration_object, optimizer, path_output: str, kwargs):
    """Run calibration for a single basin
        
    Parameters
    ----------
    training_object: dict
        Training object for each basin
    optimization_method: class
        Class with the optimization method
    path_output: str
        Path where the calibration files will be stored
    kwargs: optional
        Additional keyword arguments for the optimizer
    
    """
    optimizer_entity = optimizer(calibration_obj = calibration_object, 
                                 path_output = path_output, 
                                 kwargs = kwargs)
    optimizer_entity.run_calibration()
