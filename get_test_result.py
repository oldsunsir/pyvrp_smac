import ast
import re
import os
from multiprocessing import Pool, Lock
from functools import partial
from pyvrp import SolveParams
from params import params
from target_VRPTW import VRPTWTarget
from pyvrp.stop import MaxRuntime
from vrplib import read_solution
from StopWhenBks import StopWhenBks

# 防止打印冲突
print_lock = Lock()

bks_record = {
    "R101.100.20.vrptw" : 15720,
    "R103.100.14.vrptw" : 11680,
    "R109.100.13.vrptw" : 11090,
    "R112.100.10.vrptw" : 9080,
    "RC102.100.14.vrptw": 14190,
    "RC103.100.11.vrptw": 12210,
    "RC202.100.8.vrptw" : 10690,
    "RC203.100.5.vrptw" : 8970,
    "RC204.100.4.vrptw" : 7630,
    "RC206.100.7.vrptw" : 10240,
}
def process_instance(item : str, folder_path : str, sol_folder : str, test_pap : list[params], pattern):
    for i in range(len(test_pap)):
        test_pap[i] = params(**(test_pap[i]))
    test_pap.append(params.get_initial_params(path = "vrptw.toml"))

    instance_path = os.path.join(folder_path, item)
    single_vrp_target = VRPTWTarget(file_path = instance_path)

    max_runtime = 50

    bks = bks_record[item]
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
        # tmp_cost = single_vrp_target.model.solve(params = solve_params, 
        #             stop = MaxRuntime(max_runtime = max_runtime), seed = 42, display = False).cost()
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
        # if tmp_cost < best_res:
        #     best_res = tmp_cost
        #     best_idx = i
    with print_lock:
        print(item)
        if min_runtime == float("inf"):
            print(  f"耗时{max_runtime}s 找到可行解{best_res}\n"
                f"最佳配置 : {test_pap[best_idx].to_dict}\n")
        else:
            print(f"耗时{min_runtime}s 找到最优解{best_res}\n"
                    f"最佳配置 : {test_pap[best_idx].to_dict}\n")
        

if __name__ == "__main__":
    test_pap = []
    # max_runtime = 100

    with open("record.txt", 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("第"):
                config_text = re.search(r"第.个算法配置为(.+)", line).group(1)
                config_dict = ast.literal_eval(config_text)
                test_pap.append(config_dict)

    
    # folder_path = "Big_Test_CVRP_Data"
    folder_path = "Test_VRPTW_Data"
    sol_folder = "X"
    instances_path = os.listdir(folder_path)
    instances_path = sorted(instances_path)
    pattern = r'X-n\d+-k\d+'

    pool = Pool()
    process_func = partial(process_instance, folder_path = folder_path, sol_folder = sol_folder, test_pap = test_pap, pattern = pattern)
    pool.map(process_func, instances_path)

    pool.close()
    pool.join()