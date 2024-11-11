# Import necessary packages
from typing import Dict, List, Tuple

import numpy as np
import scipy
from hymc.modelzoo.baseconceptualmodel import BaseConceptualModel


class HBV(BaseConceptualModel):
    """HBV model.

    Implementation based on Feng et al. [1]_ and Seibert [2]_.

    References
    ----------
    .. [1] Feng, D., Liu, J., Lawson, K., & Shen, C. (2022). Differentiable, learnable, regionalized process-based
        models with multiphysical outputs can approach state-of-the-art hydrologic prediction accuracy. Water Resources
        Research, 58, e2022WR032404. https://doi.org/10.1029/2022WR032404
    .. [2] Seibert, J. (2005) HBV Light Version 2. User’s Manual. Department of Physical Geography and Quaternary
        Geology, Stockholm University, Stockholm
    
    """

    def __init__(self):
        super(HBV, self).__init__()
        self.name = "HBV"

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
        BETA, FC, K0, K1, K2, LP, PERC, UZL, TT, CFMAX, CFR, CWH, alpha, beta = param

        # Storages
        SNOWPACK = self._initial_states["SNOWPACK"]
        MELTWATER = self._initial_states["MELTWATER"]
        SM = self._initial_states["SM"]
        SUZ = self._initial_states["SUZ"]
        SLZ = self._initial_states["SLZ"]

        # run model for each timestep
        for i, (p, pet, temp) in enumerate(input):
            liquid_p, snow = (p, 0) if temp > TT else (0, p)

            # Snow module -----------------------------------------------------------------------------------------
            SNOWPACK = SNOWPACK + snow
            melt = CFMAX * (temp - TT)
            melt = max(melt, 0.0)
            melt = min(melt, SNOWPACK)
            MELTWATER = MELTWATER + melt
            SNOWPACK = SNOWPACK - melt
            refreezing = CFR * CFMAX * (TT - temp)
            refreezing = max(refreezing, 0.0)
            refreezing = min(refreezing, MELTWATER)
            SNOWPACK = SNOWPACK + refreezing
            MELTWATER = MELTWATER - refreezing
            tosoil = MELTWATER - (CWH * SNOWPACK)
            tosoil = max(tosoil, 0.0)
            MELTWATER = MELTWATER - tosoil

            # Soil and evaporation ---------------------------------------------------------------------------------
            soil_wetness = (SM / FC) ** BETA
            soil_wetness = min(max(soil_wetness, 0.0), 1.0)
            recharge = (liquid_p + tosoil) * soil_wetness

            SM = SM + liquid_p + tosoil - recharge
            excess = SM - FC
            excess = max(excess, 0.0)
            SM = SM - excess
            evapfactor = SM / (LP * FC)
            evapfactor = min(max(evapfactor, 0.0), 1.0)
            ETact = pet * evapfactor
            ETact = min(SM, ETact)
            SM = max(SM - ETact, 0.0)

            # Groundwater boxes -------------------------------------------------------------------------------------
            SUZ = SUZ + recharge + excess
            PERCact = min(SUZ, PERC)
            SUZ = SUZ - PERCact
            Q0 = K0 * max(SUZ - UZL, 0.0)
            SUZ = SUZ - Q0
            Q1 = K1 * SUZ
            SUZ = SUZ - Q1
            SLZ = SLZ + PERCact
            Q2 = K2 * SLZ
            SLZ = SLZ - Q2

            # Store time evolution of the internal states
            states["SNOWPACK"][i] = SNOWPACK
            states["MELTWATER"][i] = MELTWATER
            states["SM"][i] = SM
            states["SUZ"][i] = SUZ
            states["SLZ"][i] = SLZ

            # total outflow
            out[i] = Q0 + Q1 + Q2  # [mm]

        # routing method
        UH = self._gamma_routing(alpha=alpha, beta=beta, uh_len=15)
        out = self._uh_conv(discharge=out, unit_hydrograph=UH).reshape((-1, 1))

        return out, states

    def _gamma_routing(self, alpha: float, beta: float, uh_len: int = 10):
        """Unit hydrograph based on gamma function.

        Parameters
        ----------
        alpha: float
            Shape parameter of the Gamma distribution.
        beta: float
            Scale parameter of the Gamma distribution.
        uh_len: int
            Number of timesteps the unitary hydrograph will have.

        Returns
        -------
        uh : torch.Tensor
            Unit hydrograph
        
        """
        x = np.arange(0.5, 0.5 + uh_len, 1)
        coeff = 1 / (beta**alpha * np.exp(scipy.special.loggamma(alpha)))
        gamma_pdf = coeff * (x ** (alpha - 1)) * np.exp(-x / beta)
        # Normalize data so the sum of the pdf equals 1
        uh = gamma_pdf / np.sum(gamma_pdf)
        return uh

    def _uh_conv(self, discharge: np.ndarray, unit_hydrograph: np.ndarray):
        """Unitary hydrograph routing.

        Parameters
        ----------
        discharge:
            Discharge series
        unit_hydrograph:
            Unit hydrograph

        Returns
        -------
        y:
            Routed discharge

        """
        padding_size = unit_hydrograph.shape[0] - 1
        y = np.convolve(np.array(discharge).flatten(), unit_hydrograph, mode="full")
        return y[0:-padding_size]

    @property
    def _initial_states(self) -> Dict[str, float]:
        return {"SNOWPACK": 0.001, "MELTWATER": 0.001, "SM": 0.001, "SUZ": 0.001, "SLZ": 0.001}

    @property
    def parameter_ranges(self) -> Dict[str, List[float]]:
        return {
            "BETA": (1.0, 6.0),
            "FC": (50.0, 1000.0),
            "K0": (0.05, 0.9),
            "K1": (0.01, 0.5),
            "K2": (0.001, 0.2),
            "LP": (0.2, 1.0),
            "PERC": (0.0, 10.0),
            "UZL": (0.0, 100.0),
            "TT": (-2.5, 2.5),
            "CFMAX": (0.5, 10.0),
            "CFR": (0.0, 0.1),
            "CWH": (0.0, 0.2),
            "alpha": (0.0, 2.9),
            "beta": (0.0, 6.5),
        }
