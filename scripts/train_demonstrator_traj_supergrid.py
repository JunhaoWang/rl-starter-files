#!/usr/bin/env python3

#This script is for training the demonstrator trajectories for an environment where your goal is actually the goal of a bad expert.
#The output is the state occupancy distribution of the expert and a dictionary mapping of states to indexes

import argparse
import gym
import time
import datetime
import torch
import torch_ac
import numpy as np
import sys

import utils
from model import ACModel
from model_flat import ACModelFlat
from gym_minigrid.wrappers import FullyObsWrapper #, ReseedWrapper
import matplotlib.pyplot as plt
from utils.misc import getSSRep
from utils.aggregators import aggregateAverage, aggregateVAE
from utils.getIndexedArrayFromTrajectory import getIndexedArrayFromTrajectory, getStateIndexTraj
from utils.misc import getSSRepHelperMeta
from torch_ac.utils import DictList
#from utils.neural_density import NeuralDensity



# Parse arguments

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--nameDemonstrator", required=True,
                    help="name of the demonstrator enviorment for output (REQUIRED)")

parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 10e7)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=0,
                    help="number of updates between two saves (default: 0, 0 means no saving)")
parser.add_argument("--tb", action="store_true", default=False,
                    help="log into Tensorboard")
parser.add_argument("--frames-per-proc", type=int, default=None,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=7e-4,
                    help="learning rate for optimizers (default: 7e-4)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-5,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-5)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer apha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of timesteps gradient is backpropagated (default: 1)\nIf > 1, a LSTM is added to the model to have memory")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")
parser.add_argument("--full-obs", type=int, default=0, help="full-obs")
parser.add_argument("--flat-model", type=int, default=0, help="use flat neural architecture instead of CNN")
parser.add_argument("--arch", type=int, default=0,
                    help="architecture type, default 0, indicates the number of linear layers between the CNN and actor-critic")

parser.add_argument("--meanReward", type=float, default=0.85,
                    help="mean reward required to stop")
parser.add_argument("--rewardLB", type=float, default=0.8,
                    help="minimum reward required to stop")
args = parser.parse_args()
args.mem = args.recurrence > 1


#TODO : put that in utils
def make_dem(nb_trajs, model):
    obss = []
    trajs = []
    memory0 = torch.zeros([1,128], device = device, dtype = torch.float)
    for i in range(nb_trajs):
        done = False
        memory = memory0
        obs = env.reset()
        steps=0

        while not done:
            obs         = np.array([obs])
            obs         = torch.tensor(obs, device=device, dtype=torch.float)
            dictio = DictList({'image':obs})
            dist, value, memory = model(dictio, memory)

            #We sample an action from the distribution (stochastic policy)
            action      = dist.sample()

            #We go one step ahead
            play        = env.step(action.cpu().numpy())
            next_obs, true_reward, done = play[0], play[1], play[2]
            obss.append(np.array(obs.cpu()))
            obs = next_obs
            if done:
                obs         = np.array([obs])
                obs         = torch.tensor(obs, device=device, dtype=torch.float)
                obss.append(np.array(obs.cpu()))
                obss=np.array(obss)
                #if true_reward > 0:
                trajs.append(obss)
                obss = []
    return trajs

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

# Define run dir
## important constant
MAX_SAMPLE = 10
PERFORMANCE_THRESHOLD = args.meanReward
LB_PERFORMANCE_THRESHOLD = args.rewardLB
RECORD_OPTIMAL_TRAJ = False
OPTIMAL_TRAJ_START_IDX = -1

suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = "{}_{}_seed{}_{}".format(args.env, args.algo, args.seed, suffix)
model_name = args.model or default_model_name
model_dir = utils.get_model_dir(model_name)

# Define logger, CSV writer and Tensorboard writer

logger = utils.get_logger(model_dir)
csv_file, csv_writer = utils.get_csv_writer(model_dir)
if args.tb:
    from tensorboardX import SummaryWriter
    tb_writer = SummaryWriter(model_dir)

# Log command and all script arguments

logger.info("{}\n".format(" ".join(sys.argv)))
logger.info("{}\n".format(args))

# Set seed for all randomness sources

utils.seed(args.seed)

# Generate environments
if __name__ == '__main__':
    envs = []
    args.full_obs=1
    for i in range(args.procs):
        env = gym.make(args.env)
        if args.full_obs:
            env = FullyObsWrapper(env)
            #env=ReseedWrapper(env)
        env.seed(args.seed + 10000*i)
        envs.append(env)

    # Define obss preprocessor

    obs_space, preprocess_obss = utils.get_obss_preprocessor(args.env, envs[0].observation_space, model_dir)

    # create model params

    status = {"num_frames": 0, "update": 0}

    if(bool(args.flat_model)):
        acmodel = ACModelFlat(obs_space, envs[0].action_space, args.mem, args.text)
    else:
        print('ok')
        acmodel = ACModel(obs_space, envs[0].action_space, args.mem, args.text, args.arch)

    logger.info("Flat model successfully created\n")


    logger.info("{}\n".format(acmodel))


    # Define actor-critic model, for creating demonstrator trajectories

    if torch.cuda.is_available():
        acmodel.cuda()
    logger.info("CUDA available: {}\n".format(torch.cuda.is_available()))

    # Define actor-critic algo
    useKL=False


    if args.algo == "a2c":
        algo = torch_ac.A2CAlgo(envs, acmodel, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, preprocess_obss)
    elif args.algo == "ppo":
        algo = torch_ac.PPOAlgo(envs, acmodel, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss,
                                None,useKL)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    # Train model

    num_frames = status["num_frames"]
    total_start_time = time.time()
    update = status["update"]

    optimal_trajs=[]

    while num_frames < args.frames:
        # # visualize state representation
        # if len(acmodel.traj_info_set) > 0 and len(acmodel.traj_info_set) % 100 == 1:
        #     SSrep = getSSRep(acmodel.traj_info_set, 0, 1000, aggregator=None, method='vanilla').reshape(
        #         tuple(acmodel.traj_info_set[0][2]))
        #     plt.imshow(SSrep)
        #     plt.show()

        # Update model parameters

        update_start_time = time.time()
        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps,0,0,0)
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1

        # Print logs

        if update % args.log_interval == 0:
            fps = logs["num_frames"]/(update_end_time - update_start_time)
            duration = int(time.time() - total_start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])
            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

            logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | rR:mu sigma m M {:.2f} {:.2f} {:.2f} {:.2f} | F:mu sigma m M {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | delta {:.3f}"
                .format(*data))
            ######################################################
            # get optimal trajectory after reaching optimality
            mean_performance_lowerbound = data[4]
            if mean_performance_lowerbound > PERFORMANCE_THRESHOLD and data[6] > LB_PERFORMANCE_THRESHOLD:                
                print('agent reach optimality, start collecting trajectories')
                RECORD_OPTIMAL_TRAJ = True
                #OPTIMAL_TRAJ_START_IDX = optimal_trajs.shape[0]
                #PERFORMANCE_THRESHOLD = 100
                #TODO : check this the type you want, it will be a list of trajectories.
                #For each trajectory you will have a list of the different observations
                #Each observation is an n*n*3 tensor, n being being the size of the grid
               # trajs_after_training = make_dem(MAX_SAMPLE, acmodel)
            if RECORD_OPTIMAL_TRAJ:
                print('agent successfully collected {} trajectories'.format(MAX_SAMPLE))
                break

            ######################################################

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            if status["num_frames"] == 0:
                csv_writer.writerow(header)
            csv_writer.writerow(data)
            csv_file.flush()

            if args.tb:
                for field, value in zip(header, data):
                    tb_writer.add_scalar(field, value, num_frames)

            status = {"num_frames": num_frames, "update": update}



    #get the state occupancy distribution for the demonstrator trajectories
    print('pickling optimal trajectories')
    #if RECORD_OPTIMAL_TRAJ:
    #    import pickle
        #optimal_trajs_out=optimal_trajs[len(optimal_trajs):(optimal_trajs-10)]
        #optimal_trajs_out = optimal_trajs[OPTIMAL_TRAJ_START_IDX:OPTIMAL_TRAJ_START_IDX + MAX_SAMPLE]
    #    with open('optimal_trajs_{}.pkl'.format(args.env), 'wb') as f:
    #        pickle.dump(optimal_trajs_out, f)
    #else:
    #    raise Exception('optimality not reached')

    optimal_trajs=make_dem(1000,acmodel)
    #optimal_trajs=np.array(optimal_trajs)
    print("number of episodes sampled %s" %len(optimal_trajs))
    first=optimal_trajs[0]

    stateToIndex, indexToState = getIndexedArrayFromTrajectory(optimal_trajs[0])

    #print(stateToIndex)
    stateOccupancyList = []

    for i in range(len(optimal_trajs)):
        indexedTraj = getStateIndexTraj(optimal_trajs[i],stateToIndex, indexToState)
        stateOccupancyList.append(indexedTraj)

    #print(stateOccupancyList)

    stateOccupancyList = getSSRepHelperMeta(stateOccupancyList,len(stateToIndex),aggregateVAE,method='every')
    #print(stateOccupancyList)

    import pickle

    f= open('demonstratorSSrep_' + str(args.nameDemonstrator) +'.pkl', 'wb')
    pickle.dump(stateOccupancyList, f)

    f = open('stateToIndex_supergrid.pkl', 'wb')
    pickle.dump(stateToIndex, f)

    f = open('indexToState_supergrid.pkl', 'wb')
    pickle.dump(indexToState, f)


    if torch.cuda.is_available():
        acmodel.cpu()
    utils.save_model(acmodel, 'storage/drugAddictMode' + str(args.nameDemonstrator))
    logger.info("Model successfully saved")
    if torch.cuda.is_available():
        acmodel.cuda()

    utils.save_status(status, 'storage/drugAddictMode' + str(args.nameDemonstrator))