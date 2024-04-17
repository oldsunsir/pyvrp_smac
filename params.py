import tomli
from pyvrp import SolveParams

from pyvrp.GeneticAlgorithm import GeneticAlgorithmParams
from pyvrp.PenaltyManager import PenaltyParams
from pyvrp.Population import PopulationParams
from pyvrp.search import NeighbourhoodParams

from ConfigSpace import ConfigurationSpace,Configuration
from ConfigSpace import Integer, Categorical, Float
class params:
    """
    用于初始化某个算法的参数配置
    输入格式具体到gen,pen,pop,nb中的每个参数
    """
    def __init__(self,
                 repair_probability : float = 0.5,
                 nb_iter_no_improvement : int = 20000,

                 min_pop_size : int = 25,
                 generation_size : int = 40,
                 nb_elite : int = 4,
                 nb_close : int = 5,
                 lb_diversity : float = 0.1,
                 ub_diversity : float = 0.5,

                 weight_wait_time : float = 0,
                 weight_time_warp : float = 0,
                 nb_granular : int = 20,
                 symmetric_proximity : bool = True,
                 symmetric_neighbours : bool = True,

                 init_load_penalty : int = 20,
                 init_time_warp_penalty : int = 0,
                 repair_booster : int = 12,
                 solutions_between_updates : int = 100,
                 penalty_increase : float = 1.25,
                 penalty_decrease : float = 0.85,
                 target_feasible : float = 0.43) -> None:
        self.gen_params = GeneticAlgorithmParams(repair_probability=repair_probability,
                                                 nb_iter_no_improvement=nb_iter_no_improvement)
        self.pen_params = PenaltyParams(init_load_penalty=init_load_penalty,
                                        init_time_warp_penalty=init_time_warp_penalty,
                                        repair_booster=repair_booster,
                                        solutions_between_updates=solutions_between_updates,
                                        penalty_increase=penalty_increase,
                                        penalty_decrease=penalty_decrease,
                                        target_feasible=target_feasible)
        self.pop_params = PopulationParams(min_pop_size=min_pop_size,
                                           generation_size=generation_size,
                                           nb_elite=nb_elite,
                                           nb_close=nb_close,
                                           lb_diversity=lb_diversity,
                                           ub_diversity=ub_diversity)
        self.nb_params = NeighbourhoodParams(weight_wait_time=weight_wait_time,
                                             weight_time_warp=weight_time_warp,
                                             nb_granular=nb_granular,
                                             symmetric_proximity=symmetric_proximity,
                                             symmetric_neighbours=symmetric_neighbours)     
        
    @property
    def to_dict(self) -> dict:
        """
        获取params的字典形式
        """
        return{
            "repair_probability" : self.gen_params.repair_probability,
            "nb_iter_no_improvement" : self.gen_params.nb_iter_no_improvement,
            "min_pop_size" : self.pop_params.min_pop_size,
            "generation_size" : self.pop_params.generation_size,
            "nb_elite" : self.pop_params.nb_elite,
            "nb_close" : self.pop_params.nb_close,
            "lb_diversity" : self.pop_params.lb_diversity,
            "ub_diversity" : self.pop_params.ub_diversity,
            "weight_wait_time" : self.nb_params.weight_wait_time,
            "weight_time_warp" : self.nb_params.weight_time_warp,
            "nb_granular" : self.nb_params.nb_granular,
            "symmetric_proximity" : self.nb_params.symmetric_proximity,
            "symmetric_neighbours" : self.nb_params.symmetric_neighbours,
            "init_load_penalty" : self.pen_params.init_load_penalty,
            "init_time_warp_penalty" : self.pen_params.init_time_warp_penalty,
            "repair_booster" : self.pen_params.repair_booster,
            "solutions_between_updates" : self.pen_params.solutions_between_updates,
            "penalty_increase" : self.pen_params.penalty_increase,
            "penalty_decrease" : self.pen_params.penalty_decrease,
            "target_feasible" : self.pen_params.target_feasible
        }
    @staticmethod
    def get_configuration() -> ConfigurationSpace:
        """
        一个静态方法,用于获取整体的configuration
        """
        cs = ConfigurationSpace(seed = 0)
        repair_probability = Float(name="repair_probability",
                                   bounds=(0, 1))
        nb_iter_no_improvement = Float(name="nb_iter_no_improvement",
                                       bounds=(15000, 30000))
        
        min_pop_size = Integer(name="min_pop_size",
                               bounds=(15, 35))
        generation_size = Integer(name="generation_size",
                                  bounds=(20, 60))
        nb_elite = Integer(name="nb_elite",
                           bounds=(2, 6))
        nb_close = Integer(name="nb_close ",
                           bounds=(3, 7))
        lb_diversity = Float(name="lb_diversity",
                             bounds=(0, 0.4))
        ub_diversity = Float(name="ub_diversity",
                             bounds=(0.4, 0.8))
        
        weight_wait_time = Float(name="weight_wait_time",
                                   bounds=(0, 0.5))
        weight_time_warp = Float(name="weight_time_warp",
                                 bounds=(0.5, 1.5))
        nb_granular = Integer(name="nb_granular",
                              bounds=(15, 50))
        symmetric_proximity = Categorical(name='symmetric_proximity', 
                                          items=[True, False])
        symmetric_neighbours = Categorical(name="symmetric_neighbours",
                                           items=[True, False])

        init_load_penalty = Integer(name="init_load_penalty",
                                    bounds=(15, 30))
        init_time_warp_penalty = Integer(name="init_time_warp_penalty",
                                         bounds=(0, 10))
        repair_booster = Integer(name="repair_booster",
                                 bounds=(10, 15))
        solutions_between_updates = Integer(name="solutions_between_updates",
                                            bounds=(50, 150))
        penalty_increase = Float(name="penalty_increase",
                                 bounds=(1, 1.5))
        penalty_decrease = Float(name="penalty_decrease",
                                 bounds=(0.2, 1))
        target_feasible = Float(name="target_feasible",
                                bounds=(0.3, 0.5))
        
        cs.add_hyperparameters([repair_probability, nb_iter_no_improvement, min_pop_size,
                                generation_size, nb_elite, nb_close, lb_diversity, ub_diversity,
                                weight_wait_time, weight_time_warp, nb_granular,
                                symmetric_proximity, symmetric_neighbours,
                                init_load_penalty, init_time_warp_penalty, repair_booster, solutions_between_updates,
                                penalty_increase, penalty_decrease, target_feasible])
        return cs
        


    @classmethod
    def get_initial_params(cls, path : str):
        """
        获取初始化的参数配置
        """
        kargs = cls.params_from_file(path)
        initial_params = cls()
        initial_params.gen_params = kargs["gen_params"]
        initial_params.pen_params = kargs["pen_params"]
        initial_params.pop_params = kargs["pop_params"]
        initial_params.nb_params = kargs["nb_params"]
        return initial_params

    
    @staticmethod
    def params_from_file(path) -> dict:
        """
        从.toml文件中获取参数配置
        """
        with open(path, 'rb') as fh:
            data = tomli.load(fh)

        gen_params = GeneticAlgorithmParams(**data.get("genetic", {}))
        pen_params = PenaltyParams(**data.get("penalty", {}))
        pop_params = PopulationParams(**data.get("population", {}))
        nb_params = NeighbourhoodParams(**data.get("neighbourhood", {}))
        return {"gen_params" : gen_params, 
                "pen_params" : pen_params, 
                "pop_params" : pop_params,
                "nb_params"  : nb_params
                }
        # return SolveParams(genetic = gen_params,
        #                 penalty = pen_params,
        #                 population = pop_params,
        #                 neighbourhood = nb_params)
    
    def __str__(self):
        return (f"Genetic Algorithm Parameters: {self.gen_params}\n"
                f"Penalty Parameters: {self.pen_params}\n"
                f"Population Parameters: {self.pop_params}\n"
                f"Neighbourhood Parameters: {self.nb_params}")

    
if __name__ == "__main__":
    initial_params = params.get_initial_params("cvrp.toml")
    # print(str(initial_params))
    cs = params.get_configuration()
    print(cs["penalty_decrease"])