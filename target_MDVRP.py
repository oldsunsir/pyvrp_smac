import matplotlib.pyplot as plt
from pyvrp.stop import MaxRuntime, MaxIterations

from pyvrp import Model
from params import params_from_file


def getModel(file_path) -> Model:
    with open(file_path, 'r', encoding='utf-8') as f:
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
            x, y = int(line[1]), int(line[2])
            demand = int(line[4])
            COORDS.append((x, y))
            DEMANDS.append(demand)
        for _ in range(depotNum):
            line = f.readline().strip().split()
            x, y = int(line[1]), int(line[2])
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
            demand=DEMANDS[idx]
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

def getCost(model : Model, config_path : str):
    params = params_from_file(config_path)
    result = model.solve(stop=MaxIterations(10), seed=42, display = False, params = params)
    return result.cost()