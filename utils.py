import argparse
import sys

def parse_args():
    """
    解析命令行参数并返回结果
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--type", type = str, default = "cvrp", help = "options : cvrp mdvrp vrptw")
    parser.add_argument("-p", "--path", type = str, required = True, help = "data dir path")
    parser.add_argument("-it", "--iteration", type = int, default = 100, help = "evolution iteration for each instance")

    ## pap相关
    parser.add_argument("-k", type = int, default = 4, help = "pap capacity")
    parser.add_argument("-n", type = int, default = 10, help = "select best config every n configs")
    return parser.parse_args()

def test():
    args = parse_args()

    print(args.type, args.iteration, args.path)

if __name__ == "__main__":
    test()