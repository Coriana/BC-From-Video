# NOTE: this is _not_ the original code of IDM!
# As such, while it is close and seems to function well,
# its performance might be bit off from what is reported
# in the paper.

from argparse import ArgumentParser
import pickle
import numpy as np
import torch as th
import cv2
from inverse_dynamics_model import IDMAgent
from halo import Halo
import time

MESSAGE = """
This script will take a video, predict actions for its frames and
and write them in a .IDML file (Inverse Dynamics Model Label file)
"""

# Matches a number in the MineRL Java code regarding sensitivity
# This is for mapping from recorded sensitivity to the one used in the model
CAMERA_SCALER = 360.0 / 2400.0


def main(video):
    model = '4x_idm.model'
    weights = '4x_idm.weights'
    print(MESSAGE)
    cap = cv2.VideoCapture(video)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #print("---Frames to process:" + str(frame_count) +"---")

    agent_parameters = pickle.load(open(model, "rb"))
    print(agent_parameters)
    net_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = IDMAgent(idm_net_kwargs=net_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    FramesPerBatch = 128
    Batchs = frame_count/FramesPerBatch

    required_resolution = resolution=[640, 360]

    print(f"---Processing video in {int(Batchs)} batches---")
    start_time = time.time()

    while(cap.isOpened()):
        out_path = video[:-4] # remove .mp4
        with open(out_path + '.idml', 'a') as f: # make .idml file and open for writing
            for _ in range(int(Batchs)):
                th.cuda.empty_cache()
                frames = []            

                for _ in range(FramesPerBatch):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    #assert frame.shape[0] == required_resolution[1] and frame.shape[1] == required_resolution[0], "Video must be of resolution {}".format(required_resolution)
                    # BGR -> RGB
                    frames.append(frame[..., ::-1])
                frames = np.stack(frames)
                predicted_actions = agent.predict_actions(frames)
                for i in range(FramesPerBatch):
                    prediction = '{'
                    for y, (action_name, action_array) in enumerate(predicted_actions.items()):
                        current_prediction = action_array[0, i]
                        prediction = prediction + '"' + action_name + '" : '+ str(current_prediction) + ', '
                    prediction = prediction[:-2] # remove ', '
                    prediction = prediction + "}"
                    # print(prediction)           # Test output 
                    f.write(prediction + '\n') # write to file
    print(time.time() - start_time)
if __name__ == "__main__":
    parser = ArgumentParser("Run IDM on MineRL recordings to generate .IDML files for training.")
    parser.add_argument("--video", type=str, required=True, help="Path to a .mp4 file (Minecraft recording).")

    args = parser.parse_args()

    main(args.video)
