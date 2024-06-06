This repository is a basic implement of the imitation learning algorithm. We provide a example data for testing.

<img src=demo\policy_test.gif width=100% />

## Dependencies
The code has been tested in Ubuntu 20.04.
-  torch	1.10.0	
-  matplotlib	3.7.5
-  numpy	1.23.5
-  pandas	2.0.3
-  cv2	4.10.0.82
-  transforms3d 0.4.1

## Description
- `dataset`
  - `data_init.csv` -> Initial data for policy network training
  - `demonstrationdata.py` -> Dataloader for pre-training
  - `imitationdata.py` -> Dataloader for policy network training
- `demo`
  - `demo_demonstration.png` -> Demonstration data images
  - `policy_test.csv` -> Generated robot behavior data
- `models`
  - `behaviorcost.py` -> Pre-training network models
  - `policynet.py` -> Policy network model
- `param` -> Model parameter files
- `utils`
  - `getOutline.py` -> Calculate the forward kinematics of robots
  - `show_outline_single_fig.py` -> Displays the trajectory of the generated data
- `train_policy.py` -> Training policy network
