from hymc.datasetzoo.basin import Basin
from hymc.datasetzoo.basin_camelsde import BasinCamelsDE
from hymc.datasetzoo.basin_camelsgb import BasinCamelsGB
from hymc.datasetzoo.basin_camelsus import BasinCamelsUS
from hymc.datasetzoo.basin_camelsde_1h import BasinCamelsDE_1h


def get_dataset(dataset: str) -> Basin:
    """Get dataset, depending on the run configuration.

    Parameters
    ----------
    dataset : str
        Name of the dataset to use

    Returns
    -------
    Basin
        The dataset class
    
    """
    if dataset.lower() == "camels_us":
        return BasinCamelsUS
    elif dataset.lower() == "camels_gb":
        return BasinCamelsGB
    elif dataset.lower() == "camels_de":
        return BasinCamelsDE
    elif dataset.lower() == "camels_de_1h":
        return BasinCamelsDE_1h
    else:
        raise NotImplementedError(f"No dataset implemented for {dataset}")
