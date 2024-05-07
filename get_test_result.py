import ast
import re
import os
from pyvrp import SolveParams
from params import params
from target_CVRP import CVRPTarget
from target_VRPTW import VRPTWTarget
from target_MDVRP import MDVRPTarget
from pyvrp.stop import MaxRuntime
from vrplib import read_solution
from StopWhenBks import StopWhenBks
test_pap : list[params] = []
max_runtime = 100

with open("record.txt", 'r', encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith("第"):
            config_text = re.search(r"第.个算法配置为(.+)", line).group(1)
            config_dict = ast.literal_eval(config_text)
            test_pap.append(params(**(config_dict)))

test_pap.append(params.get_initial_params(path = "cvrp.toml"))
folder_path = "Big_Test_CVRP_Data"
sol_folder = "X"
instances_path = os.listdir(folder_path)
instances_path = sorted(instances_path)
pattern = r'X-n\d+-k\d+'
# 遍历文件夹下的所有文件和文件夹
for item in instances_path:
    instance_path = os.path.join(folder_path, item)

    single_vrp_target = CVRPTarget(file_path = instance_path)

    matches = re.findall(pattern, item)[0]
    sol_path_name = os.path.join(sol_folder, f"{matches}.sol")
    bks = read_solution(sol_path_name)["cost"]
    best_res = float("inf")
    best_idx = -1
    min_runtime = float("inf")

    for i, config in enumerate(test_pap):
        solve_params = SolveParams(genetic = config.gen_params,
                                   penalty = config.pen_params,
                                   population = config.pop_params,
                                   neighbourhood = config.nb_params)
        tmp_cost = single_vrp_target.model.solve(params = solve_params, 
                    stop = StopWhenBks(bks = bks, maxruntime = max_runtime), seed = 42, display = False)
        
        if tmp_cost.runtime < max_runtime:
            if tmp_cost.runtime < min_runtime:
                best_res = tmp_cost.cost()
                min_runtime = tmp_cost.runtime
                best_idx = i
        else:
            if min_runtime == float("inf"):
                if tmp_cost.cost() < best_res:
                    best_res = tmp_cost.cost()
                    best_idx = i
    
    print(item)
    if min_runtime == float("inf"):
        print(f"规定时间内未找到最优解\n"
                f"耗时{max_runtime}s 找到可行解{best_res}\n"
                f"最佳配置 : {test_pap[best_idx].to_dict}\n")
    else:
        print(f"耗时{min_runtime}s 找到最优解{best_res}\n"
                f"最佳配置 : {test_pap[best_idx].to_dict}\n")