from vrplib import read_solution

from pyvrp import Model, read
from pyvrp.plotting import (
    plot_coordinates,
    plot_instance,
    plot_result,
    plot_route_schedule,
)
from pyvrp.stop import MaxIterations, MaxRuntime
from params import params_from_file

def getModel(file_path) -> Model:
    INSTANCE = read(file_path, round_func="trunc1", instance_format="solomon")
    model = Model.from_data(INSTANCE)
    return model



def getCost(model : Model, config_path : str):
    params = params_from_file(config_path)
    result = model.solve(stop=MaxIterations(2000), seed=42, display = False, params = params)
    return result.cost()
