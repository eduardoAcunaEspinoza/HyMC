# Import necessary packages
from typing import Dict, List, Tuple

import numpy as np
from hymc.modelzoo.baseconceptualmodel import BaseConceptualModel


class Bucket(BaseConceptualModel):
    """Model with a single linear reservoir [#]_.
    
    References
    ----------
    .. [#] Acuña Espinoza, E., Loritz, R., Álvarez Chaves, M., Bäuerle, N., and Ehret, U.: To Bucket or not to Bucket? 
    Analyzing the performance and interpretability of hybrid hydrological models with dynamic parameterization, 
    Hydrology and Earth System Sciences, 28, 2705–2719, https://doi.org/10.5194/hess-28-2705-2024, 2024.
    
    """

    def __init__(self):
        super(Bucket, self).__init__()
        self.name = "Bucket"

    def run_model(self, input: np.ndarray, param: List[float]) -> Tuple:
        """Run the model

        Parameters
        ----------
        input : np.ndarray
            Inputs for the conceptual model
        param : List[float]
            Parameters of the model

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - out: np.ndarray
                outputs of the conceptual model
            - states: np.ndarray
                time evolution of the internal states (buckets) of the conceptual model
        
        """
        # initialize structures to store the information
        out, states = self._initialize_information(conceptual_inputs=input)
        # read parameters
        aux_ET, ki = param
        # Storages
        si = self._initial_states["si"]

        # run model for each timestep
        for i, (p, pet, _) in enumerate(input):
            # 1 bucket reservoir ------------------
            si = si + p  # [mm]
            ret = pet * aux_ET  # [mm]
            si = max(0.0, si - ret)  # [mm]
            qi_out = si / ki  # [mm]
            si = si - qi_out  # [mm]
            # Store time evolution of the internal states
            states["si"][i] = si
            # total outflow
            out[i] = qi_out  # [mm]

        return out, states

    @property
    def _initial_states(self) -> Dict[str, float]:
        return {"si": 5.0}

    @property
    def parameter_ranges(self) -> Dict[str, List[float]]:
        return {"aux_ET": (0.0, 1.5), "ki": (1.0, 500.0)}