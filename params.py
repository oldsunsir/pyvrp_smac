import tomli
from pyvrp import SolveParams

from pyvrp.GeneticAlgorithm import GeneticAlgorithmParams
from pyvrp.PenaltyManager import PenaltyParams
from pyvrp.Population import PopulationParams
from pyvrp.search import NeighbourhoodParams

class params:
    def __init__(self,
                 gen_params = GeneticAlgorithmParams(),
                 pen_params = PenaltyParams(),
                 pop_params = PopulationParams(),
                 nb_params = NeighbourhoodParams()) -> None:
        self.gen_params = gen_params
        self.pen_params = pen_params
        self.pop_params = pop_params
        self.nb_params = nb_params       

    @classmethod
    def get_initial_params(cls, path : str) -> dict:
        args = cls.params_from_file(path)
        return cls(*args)
    
    @staticmethod
    def params_from_file(path) -> dict:
        with open(path, 'rb') as fh:
            data = tomli.load(fh)

        gen_params = GeneticAlgorithmParams(**data.get("genetic", {}))
        pen_params = PenaltyParams(**data.get("penalty", {}))
        pop_params = PopulationParams(**data.get("population", {}))
        nb_params = NeighbourhoodParams(**data.get("neighbourhood", {}))
        return [gen_params, pen_params, pop_params, nb_params]
        # return SolveParams(genetic = gen_params,
        #                 penalty = pen_params,
        #                 population = pop_params,
        #                 neighbourhood = nb_params)

    @property
    def genetic(self):
        return self.gen_params

    @property
    def penalty(self):
        return self.pen_params

    @property
    def population(self):
        return self.pop_params

    @property
    def neighbourhood(self):
        return self.neighbourhood
    
    def __str__(self):
        return (f"Genetic Algorithm Parameters: {self.gen_params}\n"
                f"Penalty Parameters: {self.pen_params}\n"
                f"Population Parameters: {self.pop_params}\n"
                f"Neighbourhood Parameters: {self.nb_params}")

    
if __name__ == "__main__":
    initial_params = params.get_initial_params("cvrp.toml")
    print(str(initial_params))