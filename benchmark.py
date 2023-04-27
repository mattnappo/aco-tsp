import subprocess
from os import system
import os
from datetime import datetime
from collections import defaultdict

# Configuration
OMP_NUM_THREADS = 16
NUM_TRIALS      = 3
NUM_ITERS       = 1
ALPHA           = 0.5
BETA            = 0.2
RHO             = 0.7
DEBUG           = False
datasets        = ["ts11", "dj38", "qa194", "zi929"]
#colonies        = [1024, 2048, 4096]
colonies        = [1, 2, 3]
binaries        = ["./cpu", "./cpu_omp", "./gpu"]


# Load environment variables
env = dict(os.environ, OMP_NUM_THREADS=str(OMP_NUM_THREADS))

# Prepare files
files = [(f"./data/{file}.tsp", f"./sols/{file}.sol") for file in datasets]
debug = str(DEBUG).lower()
system("mkdir -p logs/")
save_file = datetime.now().strftime("logs/run_%d-%m-%Y_%H%M%S.txt")

# Construct commands
commands = []
for binary in binaries:
    for file, soln in files:
        for colony_size in colonies:
            for i in range(NUM_TRIALS):
                cmd = [
                    binary,
                    file,
                    soln,
                    colony_size,
                    NUM_ITERS,
                    ALPHA, BETA, RHO, debug
                ]
                commands.append(list(map(lambda x : str(x), cmd)))

# Make sure commands look good
n_cmds = len(commands)
for command in commands:
    print(command)
input("Press enter to run above commands")

# Build
#system("make clean && make")

# Run the commands and parse outputs
outputs = defaultdict(list)
log = open(save_file, "a")
for i, command in enumerate(commands):
    str_cmd = ' '.join(command)
    # Parse output
    out = subprocess.check_output(command, env=env).decode('utf-8')
    lines = out.split('\n')
    time = float(lines[1][6:])
    error = float(lines[2][7:])
    run = (time, error)

    # Store output
    outputs[str_cmd].append(run)

    # Backup / write to log
    log.write(f"{str_cmd}\n{run[0]}\n{run[1]}\n\n")
    log.flush()

    print(f"Ran {i}/{n_cmds}")
log.close()

# Prepare outputs for pretty printing to align with excel sheet
print(outputs)
rows = NUM_TRIALS*len(datasets)
cols = len(colonies)

time_tables = []
error_tables = []
for binary in binaries:
    time_table  = [[0.0 for i in range(cols)] for j in range(rows)]
    error_table = [[0.0 for i in range(cols)] for j in range(rows)]
    for i, colony in enumerate(colonies):
        for j, (dataset, soln) in enumerate(files):
            key = [
                binary,
                dataset,
                soln,
                colony,
                NUM_ITERS,
                ALPHA, BETA, RHO, debug
            ]
            key = list(map(lambda x: str(x), key))
            key = ' '.join(key)
            runs = outputs[key]
            for k, run in enumerate(runs):
                time_table[j*NUM_TRIALS + k][i]  = run[0]
                error_table[j*NUM_TRIALS + k][i] = run[1]

    time_tables.append(time_table)
    error_tables.append(error_table)
    

def pprint(matrix):
    s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))


# Print the tables
for i, table in enumerate(time_tables):
    print(f"{binaries[i]} TIME TABLE")
    pprint(table)
    print("\n\n")

for i, table in enumerate(error_tables):
    print(f"{binaries[i]} ERROR TABLE")
    pprint(table)
    print("\n\n")
