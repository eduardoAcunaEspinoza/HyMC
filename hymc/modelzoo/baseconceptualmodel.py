# Import necessary packages
from typing import Dict, List, Tuple

import numpy as np


class BaseConceptualModel():
    """Abstract base model class, don't use this class for model training!

    The purpose is to have some common operations that all conceptual models will need.

    """

    def __init__(self,):
        super().__init__()

    def run_model(self, input: np.ndarray, param: List[float]) -> Tuple:
        raise NotImplementedError

    def _initialize_information(self, conceptual_inputs: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Initialize the structures to store the time evolution of the internal states and the outflow of the 
        conceptualmodel

        Parameters
        ----------
        conceptual_inputs: np.ndarray
            Inputs of the conceptual model (dynamic forcings)

        Returns
        -------
        Tuple[np.ndarray, Dict[str, np.ndarray]]
            - q_out: np.ndarray
                Array to store the outputs of the conceptual model
            - states: Dict[str, np.ndarray]
                Dictionary to store the time evolution of the internal states (buckets) of the conceptual model
        
        """
        states = {}
        # initialize dictionary to store the evolution of the states
        for name, _ in self._initial_states.items():
            states[name] = np.zeros((conceptual_inputs.shape[0], 1))

        # initialize vectors to store the evolution of the outputs
        out = np.zeros((conceptual_inputs.shape[0], 1))

        return out, states

    @property
    def _initial_states(self) -> Dict[str, float]:
        raise NotImplementedError

    @property
    def _parameter_ranges(self) -> Dict[str, List[float]]:
        raise NotImplementedError