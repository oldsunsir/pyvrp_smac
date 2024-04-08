from ConfigSpace import ConfigurationSpace,Configuration
from ConfigSpace import Integer,Categorical
from smac import BlackBoxFacade, Scenario,intensifier
from smac import Callback,runhistory
from smac.main.smbo import SMBO
from smac.main import config_selector
from smac.runhistory import TrialInfo,TrialValue,TrialKey

from params import params

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"

k = 4##算法库成员数量 4
n = 20##每次迭代smac生成的n个算法，从这n个中挑最好的 
c = 2##基算法数量
run_num = k*n##总共需要跑k*n次
n_worker = 1

PAP = target_func.PAP

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

        # print("这里是tellstart")
        # print("finish",smbo.runhistory.finished)
        # print("submit",smbo.runhistory.submitted)
        # print("tellstart结束")
        # # if smbo.runhistory.submitted % n_worker == 0:
        # #     return False
        return None
    def on_tell_end(self, smbo: SMBO, info: TrialInfo, value: TrialValue):
        print("这里是tell_end")
        print("")

        print("finish",smbo.runhistory.finished)
        print("submit",smbo.runhistory.submitted)
        for key,val in smbo.runhistory.items():
            print(key,val)
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
    
class hgs_smac:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed = 0)
        deco = Categorical("deco", 
                        ["None", "RandomRoute", "BarycentreClustering", 
                            "BarycentreQuadrant","BarycentreSwipe","RouteHistory",
                            "RandomArc","CostArc","ArcHistory","RandomPath",
                            "CostPath","PathHistory"], default="None")
        sz = Integer("sz", (50, 300), default=200)
        di = Integer("di", (1000,10000), default=10000)
        cs.add_hyperparameters([deco,sz,di])
        return cs
    
    def train(self,config:Configuration,seed:int = 0) -> float:
        arg_sz,arg_di,arg_deco = config["sz"],config["di"],config["deco"]
        print(f"目前在HGS_trials中的PAP:{PAP}")
        return target_func.train(arg_sz,arg_di,arg_deco,flag="HGS")
    
class alns_smac:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed = 0)
        deco = Categorical("deco", 
                        ["None", "RandomRoute", "BarycentreClustering", 
                            "BarycentreQuadrant","BarycentreSwipe","RouteHistory",
                            "RandomArc","CostArc","ArcHistory","RandomPath",
                            "CostPath","PathHistory"], default="None")
        sz = Integer("sz", (50, 300), default=200)
        di = Integer("di", (1000,10000), default=10000)
        cs.add_hyperparameters([deco,sz,di])
        
        return cs
    def train(self,config:Configuration,seed:int = 0) -> float:
        arg_sz,arg_di,arg_deco = config["sz"],config["di"],config["deco"]
        print(f"目前在ALNS_trials中的PAP:{PAP}")
        return target_func.train(arg_sz,arg_di,arg_deco,flag="ALNS")#flag用来指示当前要添加的是hgs or alns
    
    
if __name__ == "__main__":
 #   stop_after = n
    hgs_model = hgs_smac()
    alns_model = alns_smac()
    hgs_sce = Scenario(hgs_model.configspace,deterministic=True,n_trials=50,n_workers=n_worker,name=hgs_smacout_name,seed=1)
    alns_sce = Scenario(alns_model.configspace,deterministic=True,n_trials=50,n_workers=n_worker,name=alns_smacout_name,seed=1)
    #添加initial design
    hgs_initial_design = BlackBoxFacade.get_initial_design(scenario=hgs_sce,n_configs=5,additional_configs=[
        Configuration(configuration_space=hgs_model.configspace,values=
                       {
                        'sz' : 200,
                        'di' : 1000,
                        'deco' : 'BarycentreClustering'
                      }),
        Configuration(configuration_space=hgs_model.configspace,values=
                      {
                          'sz' : 250,
                          'di' : 5000,
                          'deco' : 'BarycentreSwipe'
                      }),
        Configuration(configuration_space=hgs_model.configspace,values=
                      {
                          'sz' : 100,
                          'di' : 7500,
                          'deco' : 'RouteHistory'
                      }),
        Configuration(configuration_space=hgs_model.configspace,values=
                      {
                          'sz' : 300,
                          'di' : 7500,
                          'deco' : 'RandomRoute'
                      }),
        Configuration(configuration_space=hgs_model.configspace,values=
                      {
                          'sz' : 150,
                          'di' : 10000,
                          'deco' : 'ArcHistory'
                      }),
        ])
    alns_initial_design = BlackBoxFacade.get_initial_design(scenario=alns_sce,n_configs=3,additional_configs=[
        Configuration(configuration_space=alns_model.configspace,values=
                      {
                          'sz' : 100,
                          'di' : 5000,
                          'deco' : 'BarycentreClustering'
                      }),
        Configuration(configuration_space=alns_model.configspace,values=
                      {
                          'sz' : 150,
                          'di' : 1000,
                          'deco' : 'RandomRoute'
                      }),
        Configuration(configuration_space=alns_model.configspace,values=
                      {
                          'sz' : 150,
                          'di' : 1000,
                          'deco' : 'None'
                      }),
    ])
    for i in range(k):
        if i % c == 0:##此时进行HGS的配置
            BlackBoxFacade(
                scenario=hgs_sce,
                target_function=hgs_model.train,
                overwrite=True if i == 0 else False,
                callbacks=[CustomCallback()],
                logging_level=999999999,
                initial_design=hgs_initial_design,
            ).optimize()

        else:##进行ALNS配置
                BlackBoxFacade(
                scenario=alns_sce,
                target_function=alns_model.train,
                overwrite=True if i == 1 else False,
                callbacks=[CustomCallback()],
                logging_level=999999999,
                initial_design=alns_initial_design,
            ).optimize()
    print(PAP)
