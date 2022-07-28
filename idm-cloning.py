# This uses the OpenAI VPT InverDynamics Agent to predic the actions on the fly.

from argparse import ArgumentParser
import pickle
import time
import numpy as np
import torch as th
import cv2
from inverse_dynamics_model import IDMAgent
from halo import Halo
import gym
import minerl

from agent import PI_HEAD_KWARGS, MineRLAgent
from data_loader import DataLoader
from lib.tree_util import tree_map


DEVICE = "cuda"

LOSS_REPORT_RATE = 100

LEARNING_RATE = 0.000181
WEIGHT_DECAY = 0.039428
#WEIGHT_DECAY = 0.00
MAX_GRAD_NORM = 5.0

MESSAGE = """
This script will take a video, predict actions for its frames and
then train a child to learn.)
"""

# Matches a number in the MineRL Java code regarding sensitivity
# This is for mapping from recorded sensitivity to the one used in the model
CAMERA_SCALER = 360.0 / 2400.0

def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs

def main(video, in_model, in_weights, out_weights):
    env = gym.make("MineRLBasaltFindCave-v0")
    model = '4x_idm.model'
    weights = '4x_idm.weights'
    print(MESSAGE)
    
    #Load IDM agent to make the predictions
    idm_agent_parameters = pickle.load(open(model, "rb"))
    idm_net_kwargs = idm_agent_parameters["model"]["args"]["net"]["args"]
    idm_pi_head_kwargs = idm_agent_parameters["model"]["args"]["pi_head_opts"]
    idm_pi_head_kwargs["temperature"] = float(idm_pi_head_kwargs["temperature"])
    IDMagent = IDMAgent(idm_net_kwargs=idm_net_kwargs, pi_head_kwargs=idm_pi_head_kwargs)
    IDMagent.load_weights(weights)

    # Load the child to learn   
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
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

    cap = cv2.VideoCapture(video)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    FramesPerBatch = 128
    Batchs = frame_count/FramesPerBatch
    required_resolution = resolution=[640, 360]
    
    print("---Frames to process:" + str(frame_count) +"---")
    print(f"---Processing video---")
    start_time = time.time()
    batches_done = 0
    
    episode_hidden_states = {}
    dummy_first = th.from_numpy(np.array((False,))).to(DEVICE)

    loss_sum = 0
    while(cap.isOpened()):
        for _ in range(int(Batchs)):
            th.cuda.empty_cache()
            frames = []            

            for _ in range(FramesPerBatch):
                ret, frame = cap.read()
                if not ret:
                    break
               # assert frame.shape[0] == required_resolution[1] and frame.shape[1] == required_resolution[0], "Video must be of resolution {}".format(required_resolution)
                # BGR -> RGB
                frames.append(frame[..., ::-1])
            frames = np.stack(frames)
            predicted_actions = IDMagent.predict_actions(frames)
            for i in range(FramesPerBatch):
                frame = frames[i]
                prediction = '{'
                for y, (action_name, action_array) in enumerate(predicted_actions.items()):
                    #current_prediction = action_array[0, i]
                    # prediction = prediction + '"' + action_name + '": '+ str(current_prediction) + ', '
                    
                    #  prediction = prediction[:-2] # remove the excess ', '
                    #  prediction = prediction + '}' # add in the close bracket

                    # print(prediction)           # Test output 
                
                    agent_action = agent._env_action_to_agent(predicted_actions, to_torch=True, check_if_null=True) # <- fails here
                    if agent_action is None:
                        # Action was null
                        continue

                    agent_obs = agent._env_obs_to_agent({"pov": frame})
                    if episode_id not in episode_hidden_states:
                        # TODO need to clean up this hidden state after worker is done with the work item.
                        #      Leaks memory, but not tooooo much at these scales (will be a problem later).
                        episode_hidden_states[episode_id] = policy.initial_state(1)
                    agent_state = episode_hidden_states[episode_id]

                    pi_distribution, v_prediction, new_agent_state = policy.get_output_for_observation(
                        agent_obs,
                        agent_state,
                        dummy_first
                    )

                    log_prob  = policy.get_logprob_of_action(pi_distribution, agent_action)

                    # Make sure we do not try to backprop through sequence
                    # (fails with current accumulation)
                    new_agent_state = tree_map(lambda x: x.detach(), new_agent_state)
                    episode_hidden_states[episode_id] = new_agent_state

                    # Finally, update the agent to increase the probability of the
                    # taken action.
                    # Remember to take mean over batch losses
                    loss = -log_prob / BATCH_SIZE
                    batch_loss += loss.item()
                    loss.backward()
            th.nn.utils.clip_grad_norm_(trainable_parameters, MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad()
            
            batches_done = batches_done + 1
            loss_sum += batch_loss
            #if batch_i % LOSS_REPORT_RATE == 0:
            time_since_start = time.time() - start_time
            print(f"Time: {time_since_start:.2f},")
            loss_sum = 0
    state_dict = policy.state_dict()
    th.save(state_dict, out_weights)

if __name__ == "__main__":
    parser = ArgumentParser("Run IDM on MineRL recordings to generate .IDML files for training.")
    parser.add_argument("--video", type=str, required=True, help="Path to a .mp4 file (Minecraft recording).")
    parser.add_argument("--in-model", required=True, type=str, help="Path to the .model file to be finetuned")
    parser.add_argument("--in-weights", required=True, type=str, help="Path to the .weights file to be finetuned")
    parser.add_argument("--out-weights", required=True, type=str, help="Path where finetuned weights will be saved")

    args = parser.parse_args()
    main(args.video, args.in_model, args.in_weights, args.out_weights)