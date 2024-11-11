# Import necessary packages
from typing import Dict, List, Tuple

import numpy as np
from hymc.modelzoo.baseconceptualmodel import BaseConceptualModel


class Nonsense(BaseConceptualModel):
    """Nonsense model [#]_.

    Hydrological model with physically non-sensical constraints: water enters the model through the
    snow reservoir, then moves through the baseflow, interflow and finally unsaturated zone reservoirs,
    in that order, before exiting the model.

    References
    ----------
    .. [#] Acuña Espinoza, E., Loritz, R., Álvarez Chaves, M., Bäuerle, N., and Ehret, U.: To Bucket or not to Bucket? 
    Analyzing the performance and interpretability of hybrid hydrological models with dynamic parameterization, 
    Hydrology and Earth System Sciences, 28, 2705–2719, https://doi.org/10.5194/hess-28-2705-2024, 2024.
    
    """

    def __init__(self):
        super(Nonsense, self).__init__()
        self.name = "Nonsense"

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
        dd, sumax, beta, ki, kb = param

        # Storages
        ss = self._initial_states["ss"]
        su = self._initial_states["su"]
        si = self._initial_states["si"]
        sb = self._initial_states["sb"]

        # run model for each timestep
        for i, (p, pet, temp) in enumerate(input):
            # Snow module --------------------------
            if temp > 0:  # if there is snowmelt
                qs_out = min(ss, dd * temp)  # snowmelt from snow reservoir
                ss = ss - qs_out  # substract snowmelt from snow reservoir
                qsp_out = qs_out + p  # flow from snowmelt and rainfall
            else:  # if the is no snowmelt
                ss = ss + p  # precipitation accumalates as snow in snow reservoir
                qsp_out = 0.0

            # Baseflow reservoir -------------------
            sb = sb + qsp_out  # [mm]
            qb_out = sb / kb  # [mm]
            sb = sb - qb_out  # [mm]

            # Interflow reservoir ------------------
            si = si + qb_out  # [mm]
            qi_out = si / ki  # [mm]
            si = si - qi_out  # [mm]

            # Unsaturated zone----------------------
            psi = (su / sumax) ** beta  # [-]
            su_temp = su + qi_out * (1 - psi)
            su = min(su_temp, sumax)
            qu_out = qi_out * psi + max(0.0, su_temp - sumax)  # [mm]

            # Evapotranspiration -------------------
            klu = 0.9  # land use correction factor [-]
            if su <= 0.0:
                ktetha = 0.0
            elif su >= 0.8 * sumax:
                ktetha = 1.0
            else:
                ktetha = su / sumax

            ret = pet * klu * ktetha  # [mm]
            su = max(0.0, su - ret)  # [mm]

            # Store time evolution of the internal states
            states["ss"][i] = ss
            states["su"][i] = su
            states["si"][i] = si
            states["sb"][i] = sb

            # total outflow
            out[i] = qu_out  # [mm]

        return out, states

    @property
    def _initial_states(self) -> Dict[str, float]:
        return {"ss": 0.0, "su": 5.0, "si": 10.0, "sb": 15.0}

    @property
    def parameter_ranges(self) -> Dict[str, List[float]]:
        return {"dd": (0.0, 10.0), "sumax": (20.0, 700.0), "beta": (1.0, 6.0), "ki": (1.0, 100.0), "kb": (10.0, 1000.0)}