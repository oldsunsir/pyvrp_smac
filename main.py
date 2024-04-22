from ConfigSpace import ConfigurationSpace,Configuration
from smac.scenario import Scenario
from smac.facade import BlackBoxFacade
from smac import Callback
from smac.main.smbo import SMBO
from smac.runhistory import TrialInfo,TrialValue
import random
import time
from params import params
from utils import parse_args
from pap import PAP

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"



class CustomCallback(Callback):
    def __init__(self, n_config : int, _pap : PAP) -> None:
        self.trials_counter = 0
        self.n_config = n_config
        self.pap = _pap

    def on_start(self, smbo:SMBO) -> None:
        print("Start now!\n")
    
    def on_tell_end(self, smbo: SMBO, info: TrialInfo, value: TrialValue):
        print("这里是tell_end\n")

        print("finish",smbo.runhistory.finished)
        print("submit",smbo.runhistory.submitted)
        for key,val in smbo.runhistory.items():
            print(key,val)

        ##下面是每n个算法后，向PAP库添加n个中最好的一个
        if smbo.runhistory.finished % self.n_config == 0:

            print(f"Evaluated {smbo.runhistory.finished} trials so far.")
            cnt = smbo.runhistory.finished // self.n_config
            print(f"Current add the {smbo._scenario.name} {cnt}th config")

            min_cost = float("inf")
            flag_id = 0
            for key in smbo.runhistory:
                if (cnt-1) * self.n_config + 1 <= key.config_id <= cnt * self.n_config:
                    if smbo.runhistory[key].cost < min_cost:
                        min_cost = smbo.runhistory[key].cost
                        flag_id = key.config_id
            best_config = smbo.runhistory.get_config(flag_id)
            best_config_dict = dict(best_config)

            assert best_config_dict is not None
            print(f"the Best of Current {self.n_config} {smbo._scenario.name} config: {best_config_dict}")
            print(f"Current incumbent value: {smbo.runhistory.get_cost(best_config)}\n")

            self.pap.papUpdate(params(**best_config_dict))

            print(f"We just triggered to stop the optimization after {smbo.runhistory.finished} {smbo._scenario.name} finished trials.")
            print(f"目前的最佳结果记录表")
            for instance_res, algo in self.pap.instance2algo.items():
                print(f"样例{instance_res.instance_path}对应的当前最佳结果为：{instance_res.cost}\
                      对应的最佳算法为{algo}")

        print("")
        print("tell_end结束\n")
        return None
    

class main:
    def __init__(self, type, path, iteration, pap_capacity, n_config) -> None:
        """
        type 控制问题类型
        path 数据路径
        iteration 控制每个实例跑多少代
        pap_capacity pap可以装几个算法
        n_config 每n次挑选一个最佳配置

        self.initial_design 为smac框架提供一个解的参考初始值
        self.callback 回调函数, 控制修改pap
        """
        assert type in ["cvrp", "mdvrp", "vrptw"],  "problem type not in cvrp, mdvrp, vrptw"
        if type == "cvrp":
            self.initial_params = params.get_initial_params("cvrp.toml")
        elif type == "vrptw":
            self.initial_params = params.get_initial_params("")
        else:
            self.initial_params = params.get_initial_params("")
            pass
        self.pap = PAP(folder_path = path, iteration = iteration, type = type)
        self.scenario = Scenario(configspace = params.get_configuration(type = type), 
                                 deterministic = True, n_trials = pap_capacity*n_config, 
                                 name = type+"smacout"+str(random.random()+time.time()), seed = 42)
        self.initial_design = BlackBoxFacade.get_initial_design(
            scenario = self.scenario,
            additional_configs=[
                Configuration(configuration_space = params.get_configuration(type = type),
                          values = {
                              "repair_probability" : self.initial_params.gen_params.repair_probability,
                              "nb_iter_no_improvement" : self.initial_params.gen_params.nb_iter_no_improvement,
                              "min_pop_size" : self.initial_params.pop_params.min_pop_size,
                              "generation_size" : self.initial_params.pop_params.generation_size,
                              "nb_elite" : self.initial_params.pop_params.nb_elite,
                              "nb_close" : self.initial_params.pop_params.nb_close,
                              "lb_diversity" : self.initial_params.pop_params.lb_diversity,
                              "ub_diversity" : self.initial_params.pop_params.ub_diversity,
                              "weight_wait_time" : self.initial_params.nb_params.weight_wait_time,
                              "weight_time_warp" : self.initial_params.nb_params.weight_time_warp,
                              "nb_granular" : self.initial_params.nb_params.nb_granular,
                              "symmetric_proximity" : self.initial_params.nb_params.symmetric_proximity,
                              "symmetric_neighbours" : self.initial_params.nb_params.symmetric_neighbours,
                              "init_load_penalty" : self.initial_params.pen_params.init_load_penalty,
                              "init_time_warp_penalty" : self.initial_params.pen_params.init_time_warp_penalty,
                              "repair_booster" : self.initial_params.pen_params.repair_booster,
                              "solutions_between_updates" : self.initial_params.pen_params.solutions_between_updates,
                              "penalty_increase" : self.initial_params.pen_params.penalty_increase,
                              "penalty_decrease" : self.initial_params.pen_params.penalty_decrease,
                              "target_feasible" : self.initial_params.pen_params.target_feasible
                    })
                ]
            )

        self.pap_capacity = pap_capacity
        self.callback = CustomCallback(n_config = n_config, _pap = self.pap)

    def target(self, config : Configuration, seed : int = 42) -> float:
        cur_params = params(**(dict(config)))
        return self.pap.papTarget(param = cur_params)
    
    def train(self):
        BlackBoxFacade(
            scenario = self.scenario,
            target_function = self.target,
            callbacks = [self.callback],
            logging_level = 9999999999,
            initial_design = self.initial_design,
        ).optimize()
        for i, algo in enumerate(self.pap.algos):
            print(f"第{i+1}个算法配置为{algo}\n")
    
if __name__ == "__main__":
    args = parse_args()
    tmp_main = main(type = args.type, path = args.path, iteration = args.iteration, pap_capacity = args.k, n_config = args.n)
    tmp_main.train()