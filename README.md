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
  git clone -b melodic https://github.com/ros-perception/vision_opencv.git
  ```
- If you use catkin_make, compile with
  ```Shell
  catkin_make -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
  ```
- For catkin tools, use
  ```Shell
  catkin config -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
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

## Usage

First, download (or train) a model to use. You can find pre-trained models [here](https://github.com/dbolya/yolact#evaluation). The default model is [yolact_base_54_800000.pth](https://drive.google.com/file/d/1UYy3dMapbH1BnmtZU4WH1zbYgOzzHHf_/view?usp=sharing). If you want to use a Yolact++ model, you'll have to install DCNv2 (see [Yolact installation instructions](https://github.com/dbolya/yolact#installation)). Note that the DCN version shipped with Yolact does currently not work with the newest Pytorch release. An updated version can be found [here](https://github.com/jinfagang/DCNv2_latest).

You can run yolact using rosrun:
```Shell
  rosrun yolact_ros yolact_ros
```

If you want to change the default parameters, e.g. the model or image topic, you can specify them:
```Shell
  rosrun yolact_ros yolact_ros _model_path:="$(rospack find yolact_ros)/scripts/yolact/weights/yolact_base_54_800000.pth" _image_topic:="/camera/color/image_raw"
```

Alternatively, you can add the node to a launch file. An example can be found in the launch folder. You can run that launch file using:
```Shell
  roslaunch yolact_ros yolact_ros.launch
```

All parameters except for the model path are dynamically reconfigurable at runtime. Either run "rqt" and select the dynamic reconfigure plugin (Plugins -> Configuration), or run rqt_reconfigure directly ("rosrun rqt_reconfigure rqt_reconfigure"). Then select "yolact_ros" from the sidebar to see the available parameters.

The following parameters are available:

| Parameter             | Description                                                         | Default                 |
|-----------------------|---------------------------------------------------------------------|-------------------------|
| image_topic           | Image topic used for subscribing                                    | /camera/color/image_raw |
| use_compressed_image  | Subscribe to compressed image topic                                 | False                   |
| publish_visualization | Publish images with detections                                      | True                    |
| publish_detections    | Publish detections as message                                       | True                    |
| display_visualization | Display window with detection image                                 | False                   |
| display_masks         | Whether or not to display masks over bounding boxes                 | True                    |
| display_bboxes        | Whether or not to display bboxes around masks                       | True                    |
| display_text          | Whether or not to display text (class [score])                      | True                    |
| display_scores        | Whether or not to display scores in addition to classes             | True                    |
| display_fps           | When displaying video, draw the FPS on the frame                    | False                   |
| score_threshold       | Detections with a score under this threshold will not be considered | 0.0                     |
| crop_masks            | If true, crop output masks with the predicted bounding box          | True                    |
| top_k                 | Further restrict the number of predictions to parse                 | 5                       |
