import pickle
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import spotpy
from hymc.modelzoo.baseconceptualmodel import BaseConceptualModel


class Basin(object):
    """Create a calibration object following the spotpy library[#]_.

    The Basin object contains the hydrological model and the data that will be used to calibrate such model. 
    Moreover, it generates the simulation and evaluation results, and evaluates the objective function.

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
    .. [#] Houska, T., Kraft, P., Chamorro-Chavez, A. and Breuer, L.: SPOTting Model Parameters Using a Ready-Made
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
        # model and parameters that will be optimized
        self.model = model
        self.params = self._init_optimization_parameters(model.parameter_ranges)

        # Store information
        self.path_data = path_data
        self.basin_id = basin_id
        self.input_variables = input_variables
        self.target_variables = target_variables
        self.warmup_period = warmup_period

        # Process time period
        if isinstance(time_period, list):
            self.time_period = time_period
        elif isinstance(time_period, Dict):
            self.time_period = time_period[basin_id]

        # Objective function for optimization
        self.obj_func = obj_func

        # Additional features
        self.additional_features = None
        if path_additional_features:
            self.additional_features = self._load_additional_features(path_additional_features)
            
        # Read the data
        self.timeseries = self._read_data()

        # Initialize vectors to do custom splitting (custom training/testing periods)
        if isinstance(time_period, list):  # no custom splitting
            self.data_split = np.full(len(self.timeseries["df"]), True, dtype=bool)
        elif isinstance(time_period, Dict):
            self.data_split = self.timeseries["df"].index.isin(self.time_period["date"])

    def parameters(self):
        """"Generate parameters for the optimization process"""
        return spotpy.parameter.generate(self.params)

    def simulation(self, x) -> np.ndarray:
        """"Return simulated values"""
        q_sim, _ = self.model.run_model(self.timeseries["inputs"], x)
        return q_sim[:, 0]

    def evaluation(self) -> np.ndarray:
        """"Return observed values"""
        return self.timeseries["target"][:, 0]

    def objectivefunction(self, simulation, evaluation) -> float:
        """"Return objective function value
        
        Parameters 
        ----------
        simulation: np.ndarray
            Simulated values
        evaluation: np.ndarray
            Observed values
        
        Returns
        -------
        float
            Objective function value 
        
        """
        evaluation = evaluation[self.warmup_period :][self.data_split[self.warmup_period :]]
        simulation = simulation[self.warmup_period :][self.data_split[self.warmup_period :]]

        # Mask nans from evaluation data
        mask_nans = ~np.isnan(evaluation)
        masked_evaluation = evaluation[mask_nans]
        masked_simulation = simulation[mask_nans]
        # Calculate value of objective function
        like = self.obj_func(masked_evaluation, masked_simulation)

        return like

    def _init_optimization_parameters(self, parameter_ranges: Dict[str, List[float]]) -> List:
        """Create a list to define the optimization parameters so spotpy recognize them correctly

        Parameters
        ----------
        parameter_ranges: Dict[str, List[float]]
            Dictionary where the keys are the name of the calibration parameters and the values are the range in which
            the parameter can vary

        Returns
        -------
        parameter_list: List
            List with the parameters that will be optimized
        
        """
        parameter_list = []
        for param_name, param_range in parameter_ranges.items():
            parameter_list.append(spotpy.parameter.Uniform(low=param_range[0], high=param_range[1], name=param_name))
        return parameter_list

    def _read_data(self) -> pd.DataFrame:
        raise NotImplementedError

    def _load_additional_features(self, path_additional_features) -> Dict[str, pd.DataFrame]:
        """Read pickle dictionary containing additional features.

        Returns
        -------
        additional_features: Dict[str, pd.DataFrame]
            Dictionary where each key is a basin and each value is a date-time indexed pandas DataFrame with the
            additional features
        
        """
        with open(path_additional_features, "rb") as file:
            additional_features = pickle.load(file)
        return additional_features

    @staticmethod
    def create_custom_periods(custom_periods: Dict) -> Dict[str, pd.DataFrame]:
        """Create a continuous date range

        Recieves a dictionary with custom periods for each basin and return a continuous date range for the given basin
        
        Parameters
        ----------
        custom_periods: Dict
            Basin indexed dictionary with custom time periods
        
        Returns
        -------
        continuous_series: Dict[str, pd.DataFrame]
            Basin-indexed dictionary where the values are custom time periods
         
        """
        with open(custom_periods, "rb") as f:
            # Load the object from the pickle file
            dict_dates= pickle.load(f)

        custom_periods = {}
        for basin_id, values in dict_dates.items():
            date_ranges = []
            for i, start_date in enumerate(values["start_dates"]):
                date_range = pd.date_range(start_date, values["end_dates"][i])
                date_ranges.append(date_range)

            continuous_series = pd.concat([pd.DataFrame(date_range, columns=["date"]) for date_range in date_ranges])
            continuous_series = continuous_series.drop_duplicates()
            continuous_series.reset_index(drop=True, inplace=True)
            custom_periods[basin_id] = continuous_series

        return custom_periods
    
    @staticmethod
    def process_df(df: pd.DataFrame,
                   basin_id: str,
                   target_variables: List[str], 
                   training_period: Union[List[str], Dict[str, pd.DataFrame]], 
                   testing_period: Union[List[str], Dict[str, pd.DataFrame]], 
                   warmup_period: int):
        """Return the training and testing dataframes for a given basin, for the training and testing periods

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe with the information of the basin
        basin_id: str
            id of the basin that will be calibrated
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
        Tuple[pd.DataFrame, pd.DataFrame]
            training and testing dataframes
        
        """

         # Training period
        if isinstance(training_period, list):
            df_training = df.loc[training_period[0] : training_period[1], target_variables][warmup_period:]
        elif isinstance(training_period, Dict):
            data_split = df.index.isin(training_period[basin_id]["date"])
            df_training = df[warmup_period:].loc[data_split[warmup_period:], target_variables]

        # Testing period
        if isinstance(testing_period, list):
            df_testing = df.loc[testing_period[0] : testing_period[1], target_variables][warmup_period:]
        elif isinstance(testing_period, Dict):
            data_split = df.index.isin(testing_period[basin_id]["date"])
            df_testing = df[warmup_period:].loc[data_split[warmup_period:], target_variables]

        return df_training, df_testing


