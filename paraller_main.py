
import argparse
import os
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reinforced Tanks Main")
    parser.add_argument("--instances", type=int, help="Number of instances to run in parallel", default=os.cpu_count())
    n_instances = parser.parse_args().instances
    command = ""
    for _ in range(n_instances):
        command += "python3 main.py & "
    command += "wait"
    os.system(command)