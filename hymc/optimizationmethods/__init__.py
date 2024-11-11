from typing import Tuple

import spotpy
from hymc.optimizationmethods.dream import DREAM
from hymc.optimizationmethods.sce import SCE
from spotpy.likelihoods import gaussianLikelihoodMeasErrorOut
from spotpy.objectivefunctions import rmse


def get_optimizer(optimizer: str) -> spotpy.algorithms:
    """Get optimizer, depending on the run configuration.

    Parameters
    ----------
    optimizer : str
        Name of the optimizer to use

    Returns
    -------
    Basin
        The dataset class
    
    """
    if optimizer.lower() == "dream":
        return DREAM
    elif optimizer.lower() == "sce":
        return SCE
    else:
        raise NotImplementedError(f"No optimizer implemented for {optimizer}")

def get_loss_function(loss_function: str) -> Tuple[callable, bool]:
    """Get loss function, depending on the run configuration.

    Parameters
    ----------
    loss_function : str
        Name of the loss function

    Returns
    -------
    Tuple
        - Objective function
        - Boolean indicating if the objective function must be maximized 
    
    """
    if loss_function.lower() == "rmse":
        return rmse, False
    elif loss_function.lower() == "gaussian_likelihood_meas_error_out":
        return gaussianLikelihoodMeasErrorOut, True
    else:
        raise NotImplementedError(f"No loss function implemented for {loss_function}")