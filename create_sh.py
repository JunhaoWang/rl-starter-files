def create_sh(RANDOM_SEED_NUMBER, KL_WEIGHT_NUMBER, REWARD_MEAN_NUMBER, REWARD_LOWER_BOUND_NUMBER):
    sh_text = """#!/bin/bash
#SBATCH -c 16
#SBATCH --mem=160G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time 4:00:00
#SBATCH --account def-bengioy
#chmod +x main_recurrent.py
#SBATCH --output="fourRooms3_seedRANDOM_SEED_NUMBER_KLKL_WEIGHT_NUMBER_MeanREWARD_MEAN_NUMBER_LbREWARD_LOWER_BOUND_NUMBER"
module load miniconda3
source $HOME/.bashrc
source activate observe
#export BABYAI_DONE_ACTIONS=1
python -u -m scripts.train_agent --env MiniGrid-FourRoomsAddict3-v0 --algo ppo --nameDemonstrator fourRoom --flat-model 0 --KLweight KL_WEIGHT_NUMBER --useKL 1 --meanReward REWARD_MEAN_NUMBER --rewardLB REWARD_LOWER_BOUND_NUMBER --seed RANDOM_SEED_NUMBER

echo 'DONE'"""

    sh_text_updated = sh_text.replace('RANDOM_SEED_NUMBER', str(RANDOM_SEED_NUMBER)).replace('KL_WEIGHT_NUMBER', str(
        KL_WEIGHT_NUMBER)).replace(
        'REWARD_LOWER_BOUND_NUMBER', str(REWARD_LOWER_BOUND_NUMBER)).replace('REWARD_MEAN_NUMBER',
                                                                             str(REWARD_MEAN_NUMBER))

    return sh_text_updated


def create_sh_batch(params):
    info_dict = {}
    for RANDOM_SEED_NUMBER, KL_WEIGHT_NUMBER, REWARD_MEAN_NUMBER, REWARD_LOWER_BOUND_NUMBER in params:
        info_dict['sh_dir/ex_seed{}_KL{}_Mean{}_Lb{}.sh'.format(RANDOM_SEED_NUMBER, KL_WEIGHT_NUMBER,
                                                                REWARD_MEAN_NUMBER,
                                                                REWARD_LOWER_BOUND_NUMBER)] = create_sh(
            RANDOM_SEED_NUMBER, KL_WEIGHT_NUMBER, REWARD_MEAN_NUMBER, REWARD_LOWER_BOUND_NUMBER)

    for filename in info_dict:
        with open(filename, 'w') as f:
            f.write(info_dict[filename])


# create_sh_batch([(1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6)])