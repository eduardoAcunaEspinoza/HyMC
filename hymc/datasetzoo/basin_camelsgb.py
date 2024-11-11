# Import necessary packages
from typing import Dict, List, Union

import pandas as pd
from hymc.datasetzoo.basin import Basin
from hymc.modelzoo.baseconceptualmodel import BaseConceptualModel


class BasinCamelsGB(Basin):
    """Class to process data from the CAMELS GB dataset [1]_ . 
    
    The class inherits from Basin to create a calibration object following the spotpy library [2]_. However here we 
    code the _read_data method, that specify how we should read the information from CAMELS GB.

    Parameters
    ----------
    model: BaseConceptualModel
        Conceptual model that will be calibrated
    path_data : str
        Path to the folder were the data is stored
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
    .. [1] Coxon, G., Addor, N., Bloomfield, J. P., Freer, J., Fry, M., Hannaford, J., Howden, N. J. K., Lane, R., 
        Lewis, M., Robinson, E. L., Wagener, T., and Woods, R.: CAMELS-GB: Hydrometeorological time series and 
        landscape attributes for 671 catchments in Great Britain, Earth Syst. Sci. Data Discuss., 
    .. [2] Houska, T., Kraft, P., Chamorro-Chavez, A. and Breuer, L.: SPOTting Model Parameters Using a Ready-Made
        Python Package, PLoS ONE, 10(12), e0145180, doi:10.1371/journal.pone.0145180, 2015
    
    """
    
    def __init__(
        self,
        model: BaseConceptualModel,
        path_data: str,
        basin_id: str,
        input_variables: List[str],
        target_variables: List[str],
        time_period: Union[List[str], Dict[str, pd.DataFrame]],
        obj_func,
        warmup_period: int = 0,
        path_additional_features: Union[str, None] = None,
    ):
        # Run the __init__ method of BaseDataset class, where the data is processed
        super(BasinCamelsGB, self).__init__(
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

        # Read information that will be used to optimize the model
        df = read_data_camelsgb(path_data = self.path_data, basin_id = self.basin_id)

        # Add additional features in case there are any
        if self.additional_features:
            df = pd.concat([df, self.additional_features[self.basin_id]], axis=1)

        # Filter for specific time period is there is any, otherwise it takes the whole time series
        if isinstance(self.time_period, list):
            df = df.loc[self.time_period[0] : self.time_period[1], :]

        # Filter for specific time period is there is any [list]. If custom time periods are used, what we do it
        # run the model for the whole period and then filter the training/testing subsets.
        df = df.loc[:, self.input_variables + self.target_variables]

        # save information
        timeseries["df"] = df
        timeseries["inputs"] = df.loc[:, self.input_variables].to_numpy()
        timeseries["target"] = df.loc[:, self.target_variables].to_numpy().reshape((-1, 1))

        return timeseries
    
    @staticmethod
    def check_basins(path_data: str,
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
            df = read_data_camelsgb(path_data=path_data, basin_id=basin)

            df_training, df_testing = Basin.process_df(df=df, 
                                                       basin_id=basin, 
                                                       target_variables= target_variables, 
                                                       training_period = training_period, 
                                                       testing_period = testing_period, 
                                                       warmup_period = warmup_period)

            if not df_training.isna().all().item() and not df_testing.isna().all().item():
                selected_basins_id.append(basin)

        return selected_basins_id


def read_data_camelsgb(path_data: str, basin_id: str) -> pd.DataFrame:
    """Read data for a specific basin from the CAMELS GB dataset

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
    df: pd.DataFrame
        Dataframe with the input data
    
    """
    path_timeseries = path_data + "/timeseries/CAMELS_GB_hydromet_timeseries_" + basin_id + "_19701001-20150930.csv"
    df = pd.read_csv(path_timeseries, index_col="date", parse_dates=["date"])

    return df
