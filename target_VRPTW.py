from vrplib import read_solution

from pyvrp import Model, read
from pyvrp.stop import MaxIterations, MaxRuntime
from params import params
from pyvrp import SolveParams

class VRPTWTarget:
    def __init__(self, iterations : int = 100, file_path : str = None) -> None:
        self.cost : float = float("inf")    ## 初始结果应该设为无穷大
        self.iterations = iterations
        self.instance_path = file_path
        self.model = Model.from_data(read(file_path, round_func="trunc1", instance_format="solomon"))

    def __hash__(self) -> int:
        """
        hash值只和样例名有关
        """
        return hash(self.instance_path)



    def getCost(self, vrptw_params : params) -> float:
        solve_params = SolveParams(genetic = vrptw_params.gen_params,
                                   penalty = vrptw_params.pen_params,
                                   population = vrptw_params.pop_params,
                                   neighbourhood = vrptw_params.nb_params)
        result = self.model.solve(stop=MaxIterations(max_iterations = self.iterations), seed=42, display = False, params = solve_params)
        return result.cost()
