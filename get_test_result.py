import ast
import re
import os
from pyvrp import SolveParams
from params import params
from target_CVRP import CVRPTarget
from target_VRPTW import VRPTWTarget
from target_MDVRP import MDVRPTarget
from pyvrp.stop import MaxRuntime
from StopWhenBks import StopWhenBks
test_pap : list[params] = []
max_runtime = 30
with open("record.txt", 'r', encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith("第"):
            config_text = re.search(r"第.个算法配置为(.+)", line).group(1)
            config_dict = ast.literal_eval(config_text)
            test_pap.append(params(**(config_dict)))

test_pap.append(params.get_initial_params(path = "cvrp.toml"))
folder_path = "Big_Test_CVRP_Data"
instances_path = os.listdir(folder_path)
# 遍历文件夹下的所有文件和文件夹
for item in instances_path:
    instance_path = os.path.join(folder_path, item)
    single_vrp_target = CVRPTarget(file_path = instance_path)
    best_res = float("inf")
    best_idx = -1
    for i, config in enumerate(test_pap):
        solve_params = SolveParams(genetic = config.gen_params,
                                   penalty = config.pen_params,
                                   population = config.pop_params,
                                   neighbourhood = config.nb_params)
        tmp_cost = single_vrp_target.model.solve(params = solve_params, 
                    stop = StopWhenBks(bks=, maxruntime = max_runtime), seed = 42, display = False).cost()
        if tmp_cost < best_res:
            best_res = tmp_cost
            best_idx = i
    print(f"best res for {instance_path} : {best_res}\
            best config : {test_pap[best_idx].to_dict}\n")