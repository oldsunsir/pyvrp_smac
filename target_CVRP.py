from vrplib import read_solution

from pyvrp import Model, read
from pyvrp.stop import MaxIterations, MaxRuntime
from params import params
from pyvrp import SolveParams

class CVRPTarget:
    """
    iterations : 控制查找代数
    file_path : 样例路径
    self.model : 从data转化成的model
    self.instance_path : 标识是哪一个instance
    """
    def __init__(self, iterations : int = 100, file_path : str = None) -> None:
        self.cost : float = float("inf")    ## 初始结果应该设为无穷大
        self.iterations = iterations
        self.instance_path = file_path
        self.model = Model.from_data(read(file_path, round_func="round"))

    
    def __hash__(self) -> int:
        """
        hash值只和样例名有关
        """
        return hash(self.instance_path)

    def getCost(self, cvrp_params : params) -> float:
        solve_params = SolveParams(genetic = cvrp_params.gen_params,
                                   penalty = cvrp_params.pen_params,
                                   population = cvrp_params.pop_params,
                                   neighbourhood = cvrp_params.nb_params)
        result = self.model.solve(stop=MaxIterations(max_iterations = self.iterations), seed=42, display = False, params = solve_params)
        return result.cost()
        
