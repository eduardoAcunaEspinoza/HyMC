import spotpy
from hymc.datasetzoo.basin import Basin


class SCE(spotpy.algorithms.sceua):
    """Shuffled Complex Evolution Algorithm (SCE-UA) [1]_

    This class is a wrapper to define the parameters and run the calibration. The actual calibration algorithm is
    implemented by the Spotpy library [2]_.

    Parameters
    ----------
    calibration_obj : Basin
        Object containing the information required to calibrate the basin
    path_output : str
        path where the calibration files will be stored
    kwargs: optional
        Additional keyword arguments for the optimizer

    References
    ----------
    .. [1] Duan, Q., Sorooshian, S., & Gupta, V. K. (1994). Optimal use of the SCE-UA global optimization method for 
    calibrating watershed models. Journal of Hydrology, 158(3), 265-284. https://doi.org/10.1016/0022-1694(94)90057-4
    .. [2] "Houska, T., Kraft, P., Chamorro-Chavez, A. and Breuer, L.: SPOTting Model Parameters Using a Ready-Made
    Python Package, PLoS ONE, 10(12), e0145180, doi:10.1371/journal.pone.0145180, 2015"
    
    """

    # the default values are based on the example provided at: https://spotpy.readthedocs.io/en/latest/Calibration_with_SCE-UA/
    def __init__(
        self,
        calibration_obj: Basin,
        path_output: str,
        kwargs,  #optional arguments
    ):
        self.name = "sce"

        # Define default values for optional parameters
        self.defaults = {
            "dbformat": "csv",
            "save_sim": False,
            "repetitions": 5000,
            "ngs": 7,
            "kstop": 3,
            "peps": 0.1,
            "pcento":  0.1,
            "max_loop_inc": None,
            "random_state": 110,
        }

        # Update defaults with any user-provided kwargs
        self.defaults.update(kwargs)

        # Assign each parameter as an attribute
        for key, value in self.defaults.items():
            setattr(self, key, value)

        file_name = path_output + calibration_obj.model.name + "_" + self.name + "_" + calibration_obj.basin_id
        super(SCE, self).__init__(
            calibration_obj, 
            dbname=file_name, 
            dbformat=self.dbformat, 
            save_sim=self.save_sim, 
            random_state=self.random_state
        )
    
    def run_calibration(self):
        """Run the SCE-UA algorithm"""
        self.sample(
            repetitions=self.repetitions,
            ngs=self.ngs,
            kstop=self.kstop,
            peps=self.peps,
            pcento=self.pcento,
            max_loop_inc=self.max_loop_inc,
        )