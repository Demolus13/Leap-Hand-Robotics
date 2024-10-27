## Welcome to the LEAP Hand ROS1 SDK - Rock, Paper, Scissor
[https://github.com/user-attachments/assets/a1c1eca8-04b9-4a89-a0da-8a8b30c01b97](https://github.com/user-attachments/assets/6dd06678-99da-40d7-90d0-5a0ee1e62b2b)
## Prerequisites
- **Operating System:** Ubuntu 20.04
- **ROS Version:** Noetic

## Install Dependencies
```bash
pip install empy==3.3.4 catkin_pkg pyyaml rospkg
pip install dynamixel_sdk numpy
```

## Creating a ROS Workspace
1. **Install** the dependencies and navigate to your ROS workspace.
2. **Create a new workspace:**
    ```bash
    cd ~/leap_hand_ws/src
    ```
3. **Clone Leap-Hand-Robotics** inside the `~/leap_hand_ws/src` directory
    ```bash
    git clone https://github.com/Demolus13/Leap-Hand-Robotics.git leap_hand --recursive
    ```
4. **Create Executable Files** inside the `~/leap_hand_ws/src/leap_hand` directory
    ```bash
    chmod +x leaphand_node.py
    chmod +x rock_paper_scissors.py
    ```

## Building the workspace
1. **Build your workspace:**
    ```bash
    cd ~/leap_hand_ws
    catkin_make
    ```
2. **Source your workspace:**
    ```bash
    source devel/setup.bash
    ```

## Connect to the Leap Hand Hardware
### To Connect
- Connect 5v power to the hand (the dynamixels should light up during boot up.)
- Connect the Micro USB cable to the hand (Do not use too many USB extensions)
- Find the USB port using [Dynamixel Wizard](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2/)

### To Launch
1. **Terminal 1**
    ```bash
    roslaunch leap_hand example.launch
    ```
2. **Terminal 2**
    ```bash
    rosrun leap_hand rock_paper_scissors.py
    ```
Finally Enjoy Playing the Game !!!

Note: We have already saved the gestures of rock, paper, and scissors in the `rock_paper_scissors.py`
```python
# Define the joint positions for rock, paper, and scissors
self.states = {
    "rock": np.array([3.1416, 4.1888, 4.5553, 4.4157, 3.1416, 4.1190, 5.1487, 4.2412, 3.1416, 4.2237, 4.7124, 4.4506, 2.6005, 1.5184, 4.6775, 4.4157]),
    "paper": np.array([3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416]),
    "scissors": np.array([3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 4.2237, 4.7124, 4.4506, 2.6005, 1.5184, 4.6775, 4.4157])
}
```

## Additional Steps: To train your own hand gesture model
1. **Capture Images for Training** using the script `capture_images.py`
    - On executing the python file the webcam will open can you can save your hand gesture by pressing Enter-Key to capture.
    - Captured images will be stored in the [`data/raw`](./hand_gesture/data/raw/)
2. **Execture preprocessing** using the script `preprocess_data.py`
    - The preprocessed .csv files will be created in [`data/processed`](./hand_gesture/data/processed/)
3. **Train the Model** using the script `train_model.py`
    - The trained model and the label encodings will be stored in [`models`](./hand_gesture/models/)
4. **Evaluating model** using the script `evaluate_model.py`
    - The webcam will open and you can perform different stored gestures to check the accuracy of the model

Follow the above steps for simulating on the Leap Hand Hardware


