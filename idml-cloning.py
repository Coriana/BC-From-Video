# Basic behavioural cloning
# Note: this uses gradient accumulation in batches of ones
#       to perform training.
#       This will fit inside even smaller GPUs (tested on 8GB one),
#       but is slow.
# NOTE: This is _not_ the original code used for VPT!
#       This is merely to illustrate how to fine-tune the models and includes
#       the processing steps used.

# This will likely be much worse than what original VPT did:
# we are not training on full sequences, but only one step at a time to save VRAM.

from argparse import ArgumentParser
import pickle
import time
import cv2
import gym
import minerl
import torch as th
import numpy as np
import json

from agent import PI_HEAD_KWARGS, MineRLAgent
from data_loader import DataLoader
from lib.tree_util import tree_map

EPOCHS = 2
# Needs to be <= number of videos
BATCH_SIZE = 8
# Ideally more than batch size to create
# variation in datasets (otherwise, you will
# get a bunch of consecutive samples)
# Decrease this (and batch_size) if you run out of memory
#N_WORKERS = 12
DEVICE = "cuda"

LOSS_REPORT_RATE = 100

LEARNING_RATE = 0.000181
#WEIGHT_DECAY = 0.039428
WEIGHT_DECAY = 0.00
MAX_GRAD_NORM = 5.0

def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs

def behavioural_cloning_train(video, in_model, in_weights, out_weights):
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)

    # To create model with the right environment.
    # All basalt environments have the same settings, so any of them works here
    env = gym.make("MineRLBasaltFindCave-v0")
    agent = MineRLAgent(env, device=DEVICE, policy_kwargs=agent_policy_kwargs, pi_head_kwargs=agent_pi_head_kwargs)
    agent.load_weights(in_weights)

    env.close()

    policy = agent.policy
    trainable_parameters = policy.parameters()

    # Parameters taken from the OpenAI VPT paper
    optimizer = th.optim.Adam(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    video_reader, frame_idx = cv2.VideoCapture(video), 0
    idml_path = video[:-4] # remove .mp4
    idml_path = idml_path + ".idml"
    idml_file = open(idml_path,'r')
    #video_reader = cv2.VideoCapture(video)
    frame_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    start_time = time.time()
    
    Batchs = frame_count/BATCH_SIZE
    # Keep track of the hidden state per episode/trajectory.
    # DataLoader provides unique id for each episode, which will
    # be different even for the same trajectory when it is loaded
    # up again
    episode_hidden_states = {}
    dummy_first = th.from_numpy(np.array((False,))).to(DEVICE)

    loss_sum = 0
    while(video_reader.isOpened()):
        for _ in range(int(Batchs)):
            for _ in range(int(BATCH_SIZE)):
                success, frame_cv = video_reader.read()  # This frame is in BGR format
                if not success: break
                idml_info = json.loads(idml_file.readline())
                frame_idx += 1
                #if frame_idx>5: break
                pov = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
                #plt.imshow(pov)
                #minerl_action = agent.get_action(dict(pov=pov))
                print(idml_info)
                batch_loss = 0

                agent_action = agent._env_action_to_agent(idml_info, to_torch=True, check_if_null=True)
                if agent_action is None:
                    # Action was null
                    continue

                agent_obs = agent._env_obs_to_agent({"pov": dict(pov=pov)})
                if frame_idx not in episode_hidden_states:
                    # TODO need to clean up this hidden state after worker is done with the work item.
                    #      Leaks memory, but not tooooo much at these scales (will be a problem later).
                    episode_hidden_states[frame_idx] = policy.initial_state(1)
                agent_state = episode_hidden_states[frame_idx]

                pi_distribution, v_prediction, new_agent_state = policy.get_output_for_observation(
                    agent_obs,
                    agent_state,
                    dummy_first
                )

                log_prob  = policy.get_logprob_of_action(pi_distribution, agent_action)

                # Make sure we do not try to backprop through sequence
                # (fails with current accumulation)
                new_agent_state = tree_map(lambda x: x.detach(), new_agent_state)
                episode_hidden_states[frame_idx] = new_agent_state

                # Finally, update the agent to increase the probability of the
                # taken action.
                # Remember to take mean over batch losses
                loss = -log_prob / BATCH_SIZE
                batch_loss += loss.item()
                loss.backward()

            th.nn.utils.clip_grad_norm_(trainable_parameters, MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad()

            loss_sum += batch_loss
            time_since_start = time.time() - start_time
            print(f"Time: {time_since_start:.2f}, Batches:, Avrg loss: {loss_sum / LOSS_REPORT_RATE:.4f}")
            loss_sum = 0

    state_dict = policy.state_dict()
    th.save(state_dict, out_weights)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to the directory containing recordings to be trained on")
    parser.add_argument("--in-model", required=True, type=str, help="Path to the .model file to be finetuned")
    parser.add_argument("--in-weights", required=True, type=str, help="Path to the .weights file to be finetuned")
    parser.add_argument("--out-weights", required=True, type=str, help="Path where finetuned weights will be saved")

    args = parser.parse_args()
    behavioural_cloning_train(args.video, args.in_model, args.in_weights, args.out_weights)
