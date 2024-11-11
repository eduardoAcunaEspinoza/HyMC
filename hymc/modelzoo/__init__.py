from hymc.modelzoo.baseconceptualmodel import BaseConceptualModel
from hymc.modelzoo.bucket import Bucket
from hymc.modelzoo.hbv import HBV
from hymc.modelzoo.nonsense import Nonsense
from hymc.modelzoo.shm import SHM


def get_model(model: str) -> BaseConceptualModel:
    """Get model, depending on the run configuration.

    Parameters
    ----------
    model : str
        Name of the model to use

    Returns
    -------
    Basin
        The dataset class
    """
    if model.lower() == "bucket":
        return Bucket
    elif model.lower() == "hbv":
        return HBV
    elif model.lower() == "nonsense":
        return Nonsense
    elif model.lower() == "shm":
        return SHM
    
    else:
        raise NotImplementedError(f"No model implemented for {model}")
