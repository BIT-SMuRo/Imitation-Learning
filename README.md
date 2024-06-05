## Dependencies
The code has been tested in Ubuntu 20.04.
1. torch	1.10.0	
2. matplotlib	3.7.5
3. numpy	1.23.5
4. pandas	2.0.3
5. cv2	4.10.0.82
6. transforms3d 0.4.1

## Description
- `dataset`
  - `data_init.csv` -> Initial data for policy network training
  - `demonstrationdata.py` -> Dataloader for pre-training
  - `imitationdata.py` -> Dataloader for policy network training
- `demo`
  - `demo_demonstration.png` -> Demonstration data images
  - `policy_test.csv` -> Generated robot behavior data
- `models`
  - `__init__.py`
  - `behaviorcost.py` ->
  - `policynet.py` ->
- `examples`
  - `getOutline.py` ->
- `utils`
  - `getOutline.py` -> Calculate the forward kinematics of robots
