from vrplib import read_solution

from pyvrp import Model
from pyvrp.stop import MaxIterations, MaxRuntime
from params import params
from pyvrp import SolveParams
from pyvrp.plotting import plot_solution
import matplotlib.pyplot as plt
class MDVRPTarget:

    def __init__(self, iterations : int = 100, file_path : str = None) -> None:
        self.cost : float = float("inf")    ## 初始结果应该设为无穷大
        self.iterations = iterations
        self.instance_path = file_path
        self.model = self.getModel()


    def __hash__(self) -> int:
        """
        hash值只和样例名有关
        """
        return hash(self.instance_path)
    

    def getCost(self, mdvrp_params : params) -> float:
        solve_params = SolveParams(genetic = mdvrp_params.gen_params,
                                   penalty = mdvrp_params.pen_params,
                                   population = mdvrp_params.pop_params,
                                   neighbourhood = mdvrp_params.nb_params)
        result = self.model.solve(stop=MaxIterations(max_iterations = self.iterations), seed=42, display = False, params = solve_params)
        return result.cost()
    
    def getModel(self) -> Model:
        with open(self.instance_path, 'r', encoding='utf-8') as f:
            firstLine = f.readline().strip().split()
            vehicleNum  =   int(firstLine[1])
            customerNum =   int(firstLine[2])
            depotNum    =   int(firstLine[3])
            for _ in range(depotNum):
                vehicleLoad = int(f.readline().strip().split()[1])
            COORDS = []
            DEMANDS = []
            for _ in range(customerNum):
                line = f.readline().strip().split()
                x, y = float(line[1]), float(line[2])
                demand = int(line[4])
                COORDS.append((x, y))
                DEMANDS.append(demand)
            for _ in range(depotNum):
                line = f.readline().strip().split()
                x, y = float(line[1]), float(line[2])
                COORDS.append((x, y))

        m = Model()
        depots = [
            m.add_depot(
                x = COORDS[idx][0],
                y = COORDS[idx][1],
            )
            for idx in range(len(COORDS)-depotNum, len(COORDS))
        ]
        for depot in depots:
            # Two vehicles at each of the depots, with maximum route durations
            # of 30.
            m.add_vehicle_type(vehicleNum, depot=depot, capacity=vehicleLoad)

        clients = [
            m.add_client(
                x=COORDS[idx][0],
                y=COORDS[idx][1],
                delivery=DEMANDS[idx]
            )
            for idx in range(len(COORDS)-depotNum)
        ]
        locations = [*depots, *clients]
        for frm_idx, frm in enumerate(locations):
            for to_idx, to in enumerate(locations):
                distance = ((frm.x - to.x)**2+(frm.y - to.y)**2)**0.5  # Manhattan
                # distance = abs(frm.x - to.x) + abs(frm.y - to.y)
                m.add_edge(frm, to, distance=distance)
        return m

if __name__ == "__main__":
    tmp_mdvrptarget = MDVRPTarget(file_path="tmp_Data/p01.txt", iterations=1000)
    tmp_mdvrptarget.getCost(mdvrp_params = params.get_initial_params("mdvrptw.toml"))