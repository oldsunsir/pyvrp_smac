from vrplib import read_solution
from typing import Union
from pyvrp import Model, read
from pyvrp.stop import MaxIterations, MaxRuntime
from params import params
from pyvrp import SolveParams
# 将int的边值修改为float
from pyvrp import Client, Depot
import math


class Edge:
    """
    Stores an edge connecting two locations.
    """

    __slots__ = ["frm", "to", "distance", "duration"]

    def __init__(
        self,
        frm: Union[Client, Depot],
        to: Union[Client, Depot],
        distance: float,
        duration: int,
    ):
        self.frm = frm
        self.to = to
        self.distance = distance
        self.duration = duration


def Euclidean_distance(position_1, position_2) -> float:
    """Computes the Euclidean distance between two points"""
    return math.sqrt((position_1[0] - position_2[0])**2 + (position_1[1] - position_2[1])**2)

class VRPTWTarget:
    def __init__(self, iterations : int = 100, file_path : str = None) -> None:
        self.cost : float = float("inf")    ## 初始结果应该设为无穷大
        self.iterations = iterations
        self.instance_path = file_path
        # data = read(file_path, round_func="dimacs")
        data = read(file_path)
        depots = data.depots()
        clients = data.clients()
        locs = depots + clients
        self.model = Model.from_data(data)
        self.model._edges =  [
                Edge(
                    frm=locs[frm],
                    to=locs[to],
                    distance=Euclidean_distance((locs[frm].x, locs[frm].y), (locs[to].x, locs[to].y)) * 10,
                    # distance=data.dist(frm, to),

                    duration=data.duration(frm, to),
                )
                for frm in range(data.num_locations)
                for to in range(data.num_locations)
        ]

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
