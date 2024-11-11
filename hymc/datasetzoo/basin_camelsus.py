# Import necessary packages
from pathlib import Path

# Import necessary packages
from typing import Dict, List, Tuple, Union

import pandas as pd
from hymc.datasetzoo.basin import Basin
from hymc.modelzoo.baseconceptualmodel import BaseConceptualModel


class BasinCamelsUS(Basin):
    """Class to process data from the CAMELS US dataset [1]_ [2]_ 

    The class inherits from Basin to create a calibration object following the spotpy library [3]_. However here we 
    code the _read_data method, that specify how we should read the information from CAMELS US.

    Parameters
    ----------
    model: BaseConceptualModel
        Conceptual model that will be calibrated
    path_data : str
        Path to the folder were the data is stored
    forcing : str
        specificy which forcing data will be used (e.g. daymet, maurer, ndlas). Only used in CAMELS_US dataset
    basin_id: str
        id of the basin that will be calibrated
    input_variables: List[str]
        Name of variables used as dynamic series input to calibrate the model
    target_variables: List[str]
        Target variable that will be used to train the model
    time_period: Union[List[str], Dict[str, pd.DataFrame]]
        Initial and final date of the time period. Can be a list, indicating the same period for all basins, or 
        a dictionary with the basin_id as key and the custom periods for each basin as the value.
    obj_func
        Objective function to be used during optimization
    warmup_period: int
        Number of timesteps (e.g. days) to warmup the internal states of the conceptual model
    path_additional_features: Union[str, None]
        Path to a pickle file containing additional features that will be used during calibration

    References
    ----------
    .. [1] A. J. Newman, M. P. Clark, K. Sampson, A. Wood, L. E. Hay, A. Bock, R. J. Viger, D. Blodgett,
        L. Brekke, J. R. Arnold, T. Hopson, and Q. Duan: Development of a large-sample watershed-scale
        hydrometeorological dataset for the contiguous USA: dataset characteristics and assessment of regional
        variability in hydrologic model performance. Hydrol. Earth Syst. Sci., 19, 209-223,
        doi:10.5194/hess-19-209-2015, 2015
    .. [2] Addor, N., Newman, A. J., Mizukami, N. and Clark, M. P.: The CAMELS data set: catchment attributes and
        meteorology for large-sample studies, Hydrol. Earth Syst. Sci., 21, 5293-5313, doi:10.5194/hess-21-5293-2017,
        2017.
    .. [3] Houska, T., Kraft, P., Chamorro-Chavez, A. and Breuer, L.: SPOTting Model Parameters Using a Ready-Made
        Python Package, PLoS ONE, 10(12), e0145180, doi:10.1371/journal.pone.0145180, 2015
    
    """
    
    def __init__(
        self,
        model: BaseConceptualModel,
        path_data: str,
        forcing: str,
        basin_id: str,
        input_variables: List[str],
        target_variables: List[str],
        time_period: Union[List[str], Dict[str, pd.DataFrame]],
        obj_func,
        warmup_period: int = 0,
        path_additional_features: Union[str, None] = None,
    ):
        # Run the __init__ method of BaseDataset class, where the data is processed
        self.forcing = forcing
        super(BasinCamelsUS, self).__init__(
            model=model,
            path_data=path_data,
            basin_id=basin_id,
            input_variables=input_variables,
            target_variables=target_variables,
            time_period=time_period,
            obj_func=obj_func,
            warmup_period=warmup_period,
            path_additional_features=path_additional_features,
        )

    def _read_data(self) -> Dict[str, pd.DataFrame]:
        # Dictionary to store the information
        timeseries = {}

        # Input data -----------------------
        df, area = read_input_camelsus(path_data=self.path_data, basin_id=self.basin_id, forcing=self.forcing)

        # Add additional features in case there are any
        if self.additional_features:
            df = pd.concat([df, self.additional_features[self.basin_id]], axis=1)

        # Filter variables of interest
        df = df.loc[:, self.input_variables]

        # Transform tmax(C) and tmin(C) into tmean(C)
        if "tmax(C)" in df.columns and "tmin(C)" in df.columns:
            df["t_mean(C)"] = (df["tmax(C)"] + df["tmin(C)"]) / 2
            df = df.drop(columns=["tmax(C)", "tmin(C)"])
            # Filtering the list using list comprehension
            self.input_variables = [item for item in self.input_variables if item not in ["tmax(C)", "tmin(C)"]]
            self.input_variables.append("t_mean(C)")

        # Target data ----------------------
        df["QObs(mm/d)"] = read_target_camelsus(path_data = self.path_data, basin_id=self.basin_id, area= area)

        # Filter for specific time period is there is any [list]. If custom time periods are used, what we do it
        # run the model for the whole period and then filter the training/testing subsets.
        if isinstance(self.time_period, list):
            df = df.loc[self.time_period[0] : self.time_period[1], :]

        # save information
        timeseries["df"] = df
        timeseries["inputs"] = df.loc[:, self.input_variables].to_numpy()
        timeseries["target"] = df.loc[:, self.target_variables].to_numpy().reshape((-1, 1))

        return timeseries
    
    @staticmethod
    def check_basins(path_data: str,
                     forcing: str,
                     basins_id: List[str],
                     target_variables: List[str],
                     training_period: Union[List[str], Dict[str, pd.DataFrame]],
                     testing_period: Union[List[str], Dict[str, pd.DataFrame]],
                     warmup_period: int = 0,) -> List[str]:
        
        """Check if the basin have target information in the periods of interest

        Parameters
        ----------
        path_data : str
            Path to the folder were the data is stored
        forcing : str
            specificy which forcing data will be used (e.g. daymet, maurer, ndlas)
        basin_id: str
            id of the basin that will be calibrated
        input_variables: List[str]
            Name of variables used as dynamic series input to calibrate the model
        target_variables: List[str]
            Target variable that will be used to train the model
        training_period: Union[List[str], Dict[str, pd.DataFrame]]
            Initial and final date of the training period. Can be a list, indicating the same period for all basins, or 
            a dictionary with the basin_id as key and the custom periods for each basin as the value.
        testing_period: Union[List[str], Dict[str, DataFrame]]
            Initial and final date of the testing period. Can be a list, indicating the same period for all basins, or
            a dictionary with the basin_id as key and the custom periods for each basin as the value.
        warmup_period: int
            Number of timesteps (e.g. days) to warmup the internal states of the conceptual model

        Returns
        -------
        selected_basins_id: List[str]
            valid basins for training and testing

        """
        selected_basins_id = []
        # Check if basin has target information in training / testing period
        for basin in basins_id:
            # Read information that will be used to optimize the model
            df, area = read_input_camelsus(path_data=path_data, basin_id=basin, forcing=forcing)
            df["QObs(mm/d)"] = read_target_camelsus(path_data=path_data, basin_id=basin, area=area)

            df_training, df_testing = Basin.process_df(df=df, 
                                                       basin_id=basin, 
                                                       target_variables= target_variables, 
                                                       training_period = training_period, 
                                                       testing_period = testing_period, 
                                                       warmup_period = warmup_period)

            if not df_training.isna().all().item() and not df_testing.isna().all().item():
                selected_basins_id.append(basin)

        return selected_basins_id


def read_input_camelsus(path_data: str, basin_id: str, forcing: str) -> Tuple[pd.DataFrame, int]:
    """Read input data for a specific basin from the CAMELS US dataset

    Parameters
    ----------
    path_data : str
        Path to the folder were the data is stored
    basin_id: str
        id of the basin that will be calibrated
    forcing: str
        specificy which forcing data will be used (e.g. daymet, maurer, ndlas)

    Returns
    -------
    df : pd.DataFrame
        Dataframe with the catchments` timeseries
    area : int
        Catchment area (m2), specified in the header of the forcing file.
    
    """
    forcing_path = Path(path_data) / "basin_mean_forcing" / forcing
    file_path = list(forcing_path.glob(f"**/{basin_id}_*_forcing_leap.txt"))
    file_path = file_path[0]
    # Read dataframe
    with open(file_path, "r") as fp:
        # load area from header
        fp.readline()
        fp.readline()
        area = int(fp.readline())
        # load the dataframe from the rest of the stream
        df = pd.read_csv(fp, sep=r"\s+")
        df["date"] = pd.to_datetime(
            df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str), format="%Y/%m/%d"
        )
        df = df.set_index("date")

    return df, area

def read_target_camelsus(path_data: str, basin_id: str, area:float) -> pd.DataFrame:
    """Read target data for a specific basin from the CAMELS US dataset

    Parameters
    ----------
    path_data : str
        Path to the folder were the data is stored
    basin_id: str
        id of the basin that will be calibrated
    area: float
        Area of the basin

    Returns
    -------
    df: pd.Series
        Time-index pandas.Series of the discharge values (mm/day)
    
    """
    streamflow_path = Path(path_data) / "usgs_streamflow"
    file_path = list(streamflow_path.glob(f"**/{basin_id}_streamflow_qc.txt"))
    file_path = file_path[0]

    col_names = ["basin", "Year", "Mnth", "Day", "QObs", "flag"]
    df = pd.read_csv(file_path, sep=r"\s+", header=None, names=col_names)
    df["date"] = pd.to_datetime(
        df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str), format="%Y/%m/%d"
    )
    df = df.set_index("date")

    # normalize discharge from cubic feet per second to mm per day
    df.QObs = 28316846.592 * df.QObs * 86400 / (area * 10**6)

    return df.QObs