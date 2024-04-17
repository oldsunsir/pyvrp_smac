from ConfigSpace import ConfigurationSpace,Configuration
from ConfigSpace import Integer,Categorical
from smac import BlackBoxFacade, Scenario,intensifier
from smac import Callback,runhistory
from smac.main.smbo import SMBO
from smac.main import config_selector
from smac.runhistory import TrialInfo,TrialValue,TrialKey

from params import params
from target_CVRP import CVRPTarget

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"

k = 4##算法库成员数量 4
n = 20##每次迭代smac生成的n个算法，从这n个中挑最好的 
c = 2##基算法数量
run_num = k*n##总共需要跑k*n次
n_worker = 1

PAP = target_func.PAP

cvrp_smacout_name = "CVRP_NEW"
hgs_smacout_name = 'HGS_New'
alns_smacout_name = 'ALNS_New'
class StopCallback(Callback):
    def __init__(self,stop_after:int):
        self._stop_after = stop_after
    def on_tell_end(self, smbo: SMBO, info: TrialInfo, value: TrialValue):
        if smbo.runhistory.finished % self._stop_after == 0:
            return False
        return None
class CustomCallback(Callback):
    def __init__(self) -> None:
        self.trials_counter = 0

    def on_start(self, smbo:SMBO) -> None:
        print("Start now!")
        print("")

    def on_ask_start(self, smbo: SMBO):
        return None
    
    def on_tell_start(self, smbo:SMBO, info: TrialInfo, value: TrialValue):
        return None
    
    def on_tell_end(self, smbo: SMBO, info: TrialInfo, value: TrialValue):
        print("这里是tell_end")
        print("")

        print("finish",smbo.runhistory.finished)
        print("submit",smbo.runhistory.submitted)
        for key,val in smbo.runhistory.items():
            print(key,val)

        ##下面是每n个算法后，向PAP库添加n个中最好的一个
        if smbo.runhistory.finished % n == 0:

            print(f"Evaluated {smbo.runhistory.finished} trials so far.")
            cnt = smbo.runhistory.finished // n
            
            print("Current add the {} {}th config".format(smbo._scenario.name,cnt))
            min_cost = float("inf")
            flag_id = 0
            for key in smbo.runhistory:
                if (cnt-1)*n + 1 <= key.config_id <= cnt*n:
                    if smbo.runhistory[key].cost < min_cost:
                        min_cost = smbo.runhistory[key].cost
                        flag_id = key.config_id
            best_config = smbo.runhistory.get_config(flag_id)
            #incumbent = smbo.intensifier.get_incumbent()
            print(best_config)
            assert best_config is not None
            print(f"the Best of Current {n} {smbo._scenario.name} config: {best_config.get_dictionary()}")
            print(f"Current incumbent value: {smbo.runhistory.get_cost(best_config)}")
            print("")
            Config_dic = best_config.get_dictionary()
            assert smbo._scenario.name == hgs_smacout_name or smbo._scenario.name == alns_smacout_name
            flag = ''
            if smbo._scenario.name == hgs_smacout_name:
                flag = 'HGS'
                PAP['HGS'].append(Config_dic)
            else: 
                flag = 'ALNS'
                PAP['ALNS'].append(Config_dic)
            ##更新Best_Record
            Algo_name = target_func.dic_to_str(Config_dic,flag=flag)

            for instance in target_func.Best_Record.keys():
                if target_func.Current_Record[instance] != {}:
                    target_func.Best_Record[instance] = min(target_func.Best_Record[instance],target_func.Current_Record[instance][Algo_name].get())
                    target_func.Current_Record[instance] = {}            ##不需要保留这n个algo的信息，可以直接清除



            print(f"We just triggered to stop the optimization after {smbo.runhistory.finished} {smbo._scenario.name} finished trials.")
            print(f"目前的最佳结果记录表")
            for instance,res in target_func.Best_Record.items():
                print(f"样例{instance}对应的当前最佳结果为：{res}")
            print('')
            print(PAP)

            print("")
            print("tell_end结束")
            return False
        # if self.trials_counter == run_num:
        #     print(f"We just triggered to stop the optimization after {smbo.runhistory.finished} finished trials.")
        #     return False
        # if self.trials_counter == 1:
        #     print("开始等待")
        #     time.sleep(10)
        #     print("等待结束")
        print("")
        print("tell_end结束")
        return None
    


class cvrp_smac:
    def __init__(self) -> None:
        self.target = CVRPTarget()
        self.target.getModel()

    @property
    def configspace(self) -> ConfigurationSpace:
        return params.get_configuration()
    
    def train(self, config : Configuration, seed : int = 42) -> float:
        cur_params = params(**config)
        return self.target.getCost(cur_params)

    
    
if __name__ == "__main__":
 #   stop_after = n
    cvrpSmac = cvrp_smac()
    cvrpSce = Scenario(cvrpSmac.configspace, deterministic=True, n_trials=50, n_workers=n_worker,
                       name=cvrp_smacout_name, seed=42)
    
    ##处理initial_design
    initialParams = params.get_initial_params("cvrp.toml")
    cvrp_initial_design = BlackBoxFacade.get_initial_design(
        scenario=cvrpSce, 
        additional_configs=[
            Configuration(configuration_space = cvrpSmac.configspace,
                          values = {
                              "repair_probability" : initialParams.gen_params.repair_probability,
                              "nb_iter_no_improvement" : initialParams.gen_params.nb_iter_no_improvement,
                              "min_pop_size" : initialParams.pop_params.min_pop_size,
                              "generation_size" : initialParams.pop_params.generation_size,
                              "nb_elite" : initialParams.pop_params.nb_elite,
                              "nb_close" : initialParams.pop_params.nb_close,
                              "lb_diversity" : initialParams.pop_params.lb_diversity,
                              "ub_diversity" : initialParams.pop_params.ub_diversity,
                              "weight_wait_time" : initialParams.nb_params.weight_wait_time,
                              "weight_time_warp" : initialParams.nb_params.weight_time_warp,
                              "nb_granular" : initialParams.nb_params.nb_granular,
                              "symmetric_proximity" : initialParams.nb_params.symmetric_proximity,
                              "symmetric_neighbours" : initialParams.nb_params.symmetric_neighbours,
                              "init_load_penalty" : initialParams.pen_params.init_load_penalty,
                              "init_time_warp_penalty" : initialParams.pen_params.init_time_warp_penalty,
                              "repair_booster" : initialParams.pen_params.repair_booster,
                              "solutions_between_updates" : initialParams.pen_params.solutions_between_updates,
                              "penalty_increase" : initialParams.pen_params.penalty_increase,
                              "penalty_decrease" : initialParams.pen_params.penalty_decrease,
                              "target_feasible" : initialParams.pen_params.target_feasible
                          })
        ]
    )
    for i in range(k):
        BlackBoxFacade(
            scenario=cvrpSce,
            target_function=cvrpSmac.train(),
            callbacks=[CustomCallback()],
            logging_level=999999999,
            initial_design=cvrp_initial_design,
        ).optimize()
