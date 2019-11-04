# yolact_ros

ROS wrapper for Yolact.

## Related packages

[yolact_ros_msgs](https://github.com/Eruvae/yolact_ros_msgs): Provides messages for publishing the detection results.

## Installation

Yolact uses Python 3. If you use a ROS version built with Python 2, additional steps are necessary to run the node.

- Set up a Python 3 virtual environment.
- Install the packages required by Yolact. See the Readme on https://github.com/dbolya/yolact for details.
- Additionally, install the packages rospkg and empy in the virtual environment.
- You need to build the cv_bridge module of ROS with Python 3. I recommend using a workspace separate from other ROS packages. Clone the package to the workspace. You might need to adjust some of the following instructions depending on your Python installation.
  ```Shell
  git clone -b kinetic https://github.com/ros-perception/vision_opencv.git
  ```
- change line 11 of cv_bridge/CMakeList.txt from "find_package(Boost REQUIRED python3)" to "find_package(Boost REQUIRED python-py35)"
- If you use catkin_make, compile with
  ```Shell
  catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.5m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so
  ```
- For catkin tools, use
  ```Shell
  catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.5m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so
  catkin build
  ```
- add the following lines to the postactivate script of your virtual environment (Change the paths according to your workspace path, virtual environment and Python installation):
  ```Shell
  source $HOME/ros_python3/devel/setup.bash
  export OLD_PYTHONPATH="$PYTHONPATH"
  export PYTHONPATH="$HOME/.virtualenvs/yolact/lib/python3.6/site-packages:$PYTHONPATH"
  ```
- add the following lines to the postdeactivate script of your virtual environment:
  ```Shell
  export PYTHONPATH="$OLD_PYTHONPATH"
  ```
