import matplotlib.pyplot as plt
from vrplib import read_solution

from pyvrp import Model, read
from pyvrp.plotting import (
    plot_coordinates,
    plot_instance,
    plot_result,
    plot_route_schedule,
)
from pyvrp.stop import MaxIterations, MaxRuntime
from params import params
from pyvrp import SolveParams

class CVRPTarget:

    def __init__(self, iterations : int = 100) -> None:
        self.model : Model = None
        self.cost : float = 0
        self.iterations = iterations

    def getModel(self, file_path = 'CVRP_Data/X-n101-k25.vrp'):
        INSTANCE = read(file_path, round_func="round")
        self.model = Model.from_data(INSTANCE)
        

    def getCost(self, cvrp_params : params) -> float:
        solve_params = SolveParams(genetic = cvrp_params.gen_params,
                                   penalty = cvrp_params.pen_params,
                                   population = cvrp_params.pop_params,
                                   neighbourhood = cvrp_params.nb_params)
        result = self.model.solve(stop=MaxIterations(max_iterations = self.iterations), seed=42, display = False, params = solve_params)
        return result
        
