import numpy as np

command = 'accelerate launch scripts/train.py configs/default.yml --bin_size {bin_size} --mask_length {mask_len} --run_name "bin{bin_size}_mask{mask_len}" --wandb_mode=online'

bin_sizes = [4, 8, 16, 32, 64, 128, 512, 1024]
mask_lens = [1, 2, 4, 8, 16, 32, 64, 128]
machine_names = [
    "v3-1",
    "v3-2",
    "v3-3",
    "v3-4",
    "v3-5",
    "v2-1",
    "v2-2",
    "v2-3",
    "v2-4",
    "v2-5",
]
num_machines = np.ceil(len(bin_sizes) * len(mask_lens) / 10).astype(int)

counter = 0
m_counter = 0

with open("scripts/train.sh", "w") as f:
    f.write("#!/bin/bash\n")
    f.write('if [ "$1" = "--dryrun" ]; then\n')
    f.write("\taccelerate launch scripts/train.py configs/default.yml --dryrun\n")
    f.write(
        "\tglcoud storage cp gs://datasets-cdminix/libritts_feats.tar.gz /dev/shm/libritts\n"
    )
    f.write("\ttar -xzf /dev/shm/libritts/libritts_feats.tar.gz -C /dev/shm/libritts\n")
    f.write("\tgcloud storage cp gs://datasets-cdminix/default_config.yaml /dev/shm/\n")
    f.write("\trm /dev/shm/hf/accelerate/default_config.yaml\n")
    f.write("\tmv /dev/shm/default_config.yaml /dev/shm/hf/accelerate/\n")
    f.write("\texit\n")
    f.write("fi\n")
    for bin_size in bin_sizes:
        for mask_len in mask_lens:
            if counter % num_machines == 0:
                m_counter += 1
                f.write(f"# Machine {m_counter}\n")
                # check if command line arg specified this machine
                f.write(
                    'if [ "$1" = "--machine" ] && [ "$2" = "'
                    + machine_names[m_counter - 1]
                    + '" ]; then\n'
                )
            f.write("\t" + command.format(bin_size=bin_size, mask_len=mask_len) + "\n")
            if counter % num_machines == num_machines - 1:
                f.write("fi\n")
            counter += 1
