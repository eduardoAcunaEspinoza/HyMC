# Import necessary packages
from typing import Dict, List, Tuple

import numpy as np
from hymc.modelzoo.baseconceptualmodel import BaseConceptualModel


class SHM(BaseConceptualModel):
    """Modified version of the SHM [1]_ model presented in [2]_.

    References
    ----------
    .. [1] Ehret, U., van Pruijssen, R., Bortoli, M., Loritz, R.,  Azmi, E. and Zehe, E: Adaptive clustering: reducing
        the computational costs of distributed (hydrological) modelling by exploiting time-variable similarity among
        model elements. HESS, 24, 4389-4411, doi: 10.5194/hess-24-4389-2020, 2020
    .. [2] Acuña Espinoza, E., Loritz, R., Álvarez Chaves, M., Bäuerle, N., and Ehret, U.: To Bucket or not to Bucket?
        Analyzing the performance and interpretability of hybrid hydrological models with dynamic parameterization,
        Hydrology and Earth System Sciences, 28, 2705–2719, https://doi.org/10.5194/hess-28-2705-2024, 2024.
    
    """

    def __init__(self):
        super(SHM, self).__init__()
        self.name = "SHM"

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
        dd, f_thr, sumax, beta, perc, kf, ki, kb = param

        # Storages
        ss = self._initial_states["ss"]
        sf = self._initial_states["sf"]
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

            # Split snowmelt+rainfall into inflow to fastflow reservoir and unsaturated reservoir ------
            qf_in = max(0, qsp_out - f_thr)
            qu_in = min(qsp_out, f_thr)

            # Fastflow module ----------------------
            sf = sf + qf_in
            qf_out = sf / kf
            sf = sf - qf_out

            # Unsaturated zone----------------------
            psi = (su / sumax) ** beta  # [-]
            su_temp = su + qu_in * (1 - psi)
            su = min(su_temp, sumax)
            qu_out = qu_in * psi + max(0.0, su_temp - sumax)  # [mm]

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

            # Interflow reservoir ------------------
            qi_in = qu_out * perc  # [mm]
            si = si + qi_in  # [mm]
            qi_out = si / ki  # [mm]
            si = si - qi_out  # [mm]

            # Baseflow reservoir -------------------
            qb_in = qu_out * (1 - perc)  # [mm]
            sb = sb + qb_in  # [mm]
            qb_out = sb / kb  # [mm]
            sb = sb - qb_out  # [mm]

            # Store time evolution of the internal states
            states["ss"][i] = ss
            states["sf"][i] = sf
            states["su"][i] = su
            states["si"][i] = si
            states["sb"][i] = sb

            # total outflow
            out[i] = qf_out + qi_out + qb_out  # [mm]

        return out, states

    @property
    def _initial_states(self) -> Dict[str, float]:
        return {"ss": 0.0, "sf": 1.0, "su": 5.0, "si": 10.0, "sb": 15.0}

    @property
    def parameter_ranges(self) -> Dict[str, List[float]]:
        return {
            "dd": (0.0, 10.0),
            "f_thr": (10.0, 60.0),
            "sumax": (20.0, 700.0),
            "beta": (1.0, 6.0),
            "perc": (0.0, 1.0),
            "kf": (1.0, 20.0),
            "ki": (1.0, 100.0),
            "kb": (10.0, 1000.0),
        }