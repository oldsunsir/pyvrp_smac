from params import params
from target_CVRP import CVRPTarget
import os

class PAP:
    """
    获取一个并行算法库
    folder_path : 样例所在的文件夹路径
    self.instances : 所有样例路径
    self.algos : 算法集合
    self.instance2algo : 样例对应的最佳算法与结果, 其中key为CVRPTarge实例, val是对应算法参数字典
    self.instancesResOfParam : 每个测试的param对应的所有instance结果, 方便后续处理instance2algo中cvrptarge.cost 与 dict
                                注意需要通过回调函数即时清除, 比如我要每10个算法选一个最好的, 那就每评估10个之后清除
    """
    def __init__(self, folder_path) -> None:
        self.instances = []

        instances_path = os.listdir(folder_path)
        # 遍历文件夹下的所有文件和文件夹
        for item in instances_path:
            instance_path = os.path.join(folder_path, item)
            if os.path.isfile(instance_path):
                self.instances.append(instance_path)

        ## 初始每个instance对应一个空算法
        self.instance2algo = {CVRPTarget(file_path = instance) : dict() for instance in self.instances}

        self.algos : list[dict] = []
        self.instancesResOfParam : dict = {}

    def papTarget(self, param : params) -> float:
        """
        获取 当前算法加入PAP后, 整体样例的结果
        主要逻辑：
            针对某个样例, 对比instance2algo中记录的结果与当前param计算结果, 
            取最小值
            结果依然是越小越好
        注意需要更新self.instancesResOfParam
        """
        sum = 0
        self.instancesResOfParam[param] = {}
        for cvrp_target, _ in self.instance2algo.items():
            cur_res = cvrp_target.getCost(cvrp_params = param)
            self.instancesResOfParam[param][cvrp_target] = cur_res
            sum += min(cur_res, cvrp_target.cost)
        return sum

    def papUpdate(self, bestParam : params):
        """
        从每n个算法中, 选出最佳参数配置的算法后, 即时更新instance2algo
        """
        for cvrp_target, _ in self.instance2algo.items():
            assert bestParam in self.instancesResOfParam.keys()
            if self.instancesResOfParam[bestParam][cvrp_target] < cvrp_target.cost:
                cvrp_target.cost = self.instancesResOfParam[bestParam][cvrp_target]
                self.instance2algo[cvrp_target] = bestParam.to_dict

        


    