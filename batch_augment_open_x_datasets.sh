: '
Script for downloading, cleaning and resizing Open X-Embodiment Dataset (https://robotics-transformer-x.github.io/)

Performs the preprocessing steps:
  1. Downloads mixture of 25 Open X-Embodiment datasets
  2. Runs resize function to convert all datasets to 256x256 (if image resolution is larger) and jpeg encoding
  3. Fixes channel flip errors in a few datsets, filters success-only for QT-Opt ("kuka") data

To reduce disk memory usage during conversion, we download the datasets 1-by-1, convert them
and then delete the original.
We specify the number of parallel workers below -- the more parallel workers, the faster data conversion will run.
Adjust workers to fit the available memory of your machine, the more workers + episodes in memory, the faster.
The default values are tested with a server with ~120GB of RAM and 24 cores.
'

DOWNLOAD_DIR=/shared/projects/mirage2/octo_oxe
CONVERSION_DIR=/shared/projects/mirage2/view_aug_octo_oxe
N_WORKERS=1                  # number of workers used for parallel conversion --> adjust based on available RAM
MAX_EPISODES_IN_MEMORY=50    # number of episodes converted in parallel --> adjust based on available RAM

# increase limit on number of files opened in parallel to 20k --> conversion opens up to 1k temporary files
# in /tmp to store dataset during conversion
ulimit -n 20000

echo "!!! Warning: This script downloads the Bridge dataset from the Open X-Embodiment bucket, which is currently outdated !!!"
echo "!!! Instead download the bridge_dataset from here: https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/ !!!"

# format: [dataset_name, dataset_version, transforms]
DATASET_TRANSFORMS=(
    "fractal20220817_data 0.1.0 view_augmentation"
    "bridge 0.1.0 view_augmentation"  
    "kuka 0.1.0 view_augmentation"
    "taco_play 0.1.0 view_augmentation"
    "jaco_play 0.1.0 view_augmentation"
    "berkeley_cable_routing 0.1.0 view_augmentation"
    "roboturk 0.1.0 view_augmentation"
    "nyu_door_opening_surprising_effectiveness 0.1.0 view_augmentation"
    "viola 0.1.0 view_augmentation"
    "berkeley_autolab_ur5 0.1.0 view_augmentation"
    "toto 0.1.0 view_augmentation"
    "language_table 0.1.0 view_augmentation"
    "stanford_hydra_dataset_converted_externally_to_rlds 0.1.0 view_augmentation"
    "austin_buds_dataset_converted_externally_to_rlds 0.1.0 view_augmentation"
    "nyu_franka_play_dataset_converted_externally_to_rlds 0.1.0 view_augmentation"
    "furniture_bench_dataset_converted_externally_to_rlds 0.1.0 view_augmentation"
    "ucsd_kitchen_dataset_converted_externally_to_rlds 0.1.0 view_augmentation"
    "austin_sailor_dataset_converted_externally_to_rlds 0.1.0 view_augmentation"
    "austin_sirius_dataset_converted_externally_to_rlds 0.1.0 view_augmentation"
    "bc_z 0.1.0 view_augmentation"
    "dlr_edan_shared_control_converted_externally_to_rlds 0.1.0 view_augmentation"
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds 0.1.0 view_augmentation"
    "utaustin_mutex 0.1.0 view_augmentation"
    "berkeley_fanuc_manipulation 0.1.0 view_augmentation"
    "cmu_stretch 0.1.0 view_augmentation"
)

for tuple in "${DATASET_TRANSFORMS[@]}"; do
  # Extract strings from the tuple
  strings=($tuple)
  DATASET=${strings[0]}
  VERSION=${strings[1]}
  TRANSFORM=${strings[2]}
  mkdir ${DOWNLOAD_DIR}/${DATASET}
  python3 modify_rlds_dataset.py --dataset=$DATASET --data_dir=$DOWNLOAD_DIR --target_dir=$CONVERSION_DIR --mods=$TRANSFORM --n_workers=$N_WORKERS --max_episodes_in_memory=$MAX_EPISODES_IN_MEMORY
done
