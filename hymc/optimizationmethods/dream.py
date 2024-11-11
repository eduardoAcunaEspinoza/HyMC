import spotpy
from hymc.datasetzoo.basin import Basin


class DREAM(spotpy.algorithms.dream):
    """DiffeRential Evolution Adaptive Metropolis (DREAM) algorithhm [1]_

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
    .. [1] Vrugt, J. A. (2016). Markov chain Monte Carlo simulation using the DREAM software package: Theory, concepts,
    and MATLAB implementation. Environmental Modelling & Software, 75, 273-316. https://doi.org/10.1016/j.envsoft.2015.08.013

    .. [2] "Houska, T., Kraft, P., Chamorro-Chavez, A. and Breuer, L.: SPOTting Model Parameters Using a Ready-Made
    Python Package, PLoS ONE, 10(12), e0145180, doi:10.1371/journal.pone.0145180, 2015"
    
    """

    def __init__(
        self,
        calibration_obj: Basin,
        path_output: str,
        kwargs,  #optional arguments
    ):
        self.name = "dream"

        # Define default values for optional parameters
        self.defaults = {
            "dbformat": "csv",
            "save_sim": False,
            "repetitions": 5000,
            "nChains": 7,
            "nCr": 3,
            "delta": 3,
            "c": 0.1,
            "eps": 10e-6,
            "convergence_limit": 1.2,
            "runs_after_convergence": 100,
            "acceptance_test_option": 6,
            "random_state": 110,
        }

        # Update defaults with any user-provided kwargs
        self.defaults.update(kwargs)

        # Assign each parameter as an attribute
        for key, value in self.defaults.items():
            setattr(self, key, value)

        file_name = path_output + calibration_obj.model.name + "_" + self.name + "_" + calibration_obj.basin_id
        super(DREAM, self).__init__(
            calibration_obj, 
            dbname=file_name, 
            dbformat=self.dbformat, 
            save_sim=self.save_sim, 
            random_state=self.random_state
        )

    def run_calibration(self):
        """Run the DREAM algorithm"""
        self.sample(
            repetitions=self.repetitions,
            nChains=self.nChains,
            nCr=self.nCr,
            delta=self.delta,
            c=self.c,
            eps=self.eps,
            convergence_limit=self.convergence_limit,
            runs_after_convergence=self.runs_after_convergence,
            acceptance_test_option=self.acceptance_test_option,
        )
