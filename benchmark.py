import subprocess
from os import system
import os
from datetime import datetime

# Configuration
OMP_NUM_THREADS = 16
NUM_TRIALS      = 3
NUM_ITERS       = 1
ALPHA           = 0.5
BETA            = 0.2
RHO             = 0.7
DEBUG           = False
datasets        = ["ts11", "dj38", "qa194", "zi929"]
colonies        = [1024, 2048, 4096]
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
outputs = dict()
log = open(save_file, "a")
for i, command in enumerate(commands):
    str_cmd = ' '.join(command)
    # Parse output
    out = subprocess.check_output(command, env=env).decode('utf-8')
    lines = out.split('\n')
    time = float(lines[1][6:])
    accuracy = float(lines[2][7:])
    run = (time, accuracy)

    # Store output
    outputs[str_cmd] = run

    # Backup / write to log
    log.write(f"{str_cmd}\n{run[0]}\n{run[1]}\n\n")

    print(f"Ran {i}/{n_cmds}")
log.close()


print(outputs)

