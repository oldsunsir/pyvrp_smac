from params import params
from target_CVRP import CVRPTarget
from target_MDVRP import MDVRPTarget
from target_VRPTW import VRPTWTarget
import os

class PAP:
    """
    获取一个并行算法库
    folder_path : 样例所在的文件夹路径
    self.instances : 所有样例路径
    self.algos : 算法集合
    self.instance2algo : 样例对应的最佳算法与结果, 其中key为CVRPTarge实例, val是对应算法参数字典
    self.instance_res_of_param : 每个测试的param对应的所有instance结果, 方便后续处理instance2algo中cvrptarge.cost 与 dict
                                注意需要通过回调函数即时清除, 比如我要每10个算法选一个最好的, 那就每评估10个之后清除
    """
    def __init__(self, folder_path : str, iteration : int = 100, type : str = "cvrp") -> None:
        self.instances = []

        instances_path = os.listdir(folder_path)
        # 遍历文件夹下的所有文件和文件夹
        for item in instances_path:
            instance_path = os.path.join(folder_path, item)
            if os.path.isfile(instance_path):
                self.instances.append(instance_path)

        ## 初始每个instance对应一个空算法
        if type == "cvrp":
            self.instance2algo = {CVRPTarget(file_path = instance, iterations = iteration) : dict() for instance in self.instances}
        elif type == "mdvrp":
            self.instance2algo = {MDVRPTarget(file_path = instance, iterations = iteration) : dict() for instance in self.instances}
        else:
            self.instance2algo = {VRPTWTarget(file_path = instance, iterations = iteration) : dict() for instance in self.instances}
        self.algos : list[dict] = []
        self.instance_res_of_param : dict = {}

    def papTarget(self, param : params) -> float:
        """
        获取 当前算法加入PAP后, 整体样例的结果
        主要逻辑：
            针对某个样例, 对比instance2algo中记录的结果与当前param计算结果, 
            取最小值
            结果依然是越小越好
        注意需要更新self.instance_res_of_param
        """
        sum = 0
        self.instance_res_of_param[param] = {}
        for vrp_target, _ in self.instance2algo.items():
            cur_res = vrp_target.getCost(mdvrp_params = param)
            self.instance_res_of_param[param][vrp_target] = cur_res
            sum += min(cur_res, vrp_target.cost)
        return sum

    def papUpdate(self, best_param : params):
        """
        从每n个算法中, 选出最佳参数配置的算法后, 即时更新instance2algo与algos
        """
        best_dict_of_param = best_param.to_dict
        self.algos.append(best_dict_of_param)
        assert best_param in self.instance_res_of_param.keys()
        for vrp_target in self.instance2algo.keys():
            if self.instance_res_of_param[best_param][vrp_target] < vrp_target.cost:
                vrp_target.cost = self.instance_res_of_param[best_param][vrp_target]
                self.instance2algo[vrp_target] = best_dict_of_param
        ##每个param更新过之后就不再用到
        self.instance_res_of_param.clear()
        

if __name__ == "__main__":
    tmp_pap = PAP(folder_path = "CVRP_Data")
    tmp_param = params.get_initial_params(path = "cvrp.toml")
    tmp_pap.papTarget(param = tmp_param)
    tmp_pap.papUpdate(best_param = tmp_param)
    for cvrp_target, dict1 in tmp_pap.instance2algo.items():
        print(f"{cvrp_target.instance_path} 当前结果为 {cvrp_target.cost}")
        print(dict1)
    