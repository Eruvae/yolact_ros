# Package Summary

**ROS2** wrapper for Yolact.

**WORK IN PROGRESS**

## yolact_ros

* **Maintainer status:** maintained
* **Maintainer:** Fernando González Ramos <fergonzaramos@yahoo.es>
* **Author:** Fernando González Ramos <fergonzaramos@yahoo.es>
* **License:** BSD

### Contents

* Overview
  * Previous steps
* Quick start
  * Instalation
  * How it works
* Nodes
  * yolact_ros2_nodes
    * Topics

## Overview

*yolact_ros2* is a **ROS2 Eloquent** wrapper for neural network [yolact](https://github.com/dbolya/yolact)

### Previous Steps

The first we have to know is that *yolact_ros2* package have dependencies as follow:

* rclpy
* std_msgs
* sensor_msgs
* yolact_ros2_msgs
* cv_bridge

## Quick Start

### Installation

You must to clone *yolact_ros2* into *src* folder located in your ROS2 workspace. Later, you have to compile it typing ``colcon build``.

### How it Works

First, download (or train) a model to use. You can find pre-trained models [here](https://github.com/dbolya/yolact#evaluation). The default model is [yolact_base_54_800000.pth](https://drive.google.com/file/d/1UYy3dMapbH1BnmtZU4WH1zbYgOzzHHf_/view?usp=sharing). If you want to use a Yolact++ model, you'll have to install DCNv2 (see [Yolact installation instructions](https://github.com/dbolya/yolact#installation)). Note that the DCN version shipped with Yolact does currently not work with the newest Pytorch release. An updated version can be found [here](https://github.com/jinfagang/DCNv2_latest).

**NOTE:** Before running *yolact_ros*, make sure that camera driver is running. You can use the camera you want.

Now, you can run *yolact_ros2* typing:

```
$ ros2 launch yolact_ros2 yolact_ros2.launch.py
```

If you want to change the default parameters, e.g. the model or image topic, you can specify them editing the file named **yolact_config.yaml** located at *[your_workspace]/src/yolact_ros2/config/*. Main parameters are the following:

The following parameters are available:

| Parameter             | Description                                                         | Default                 |
|-----------------------|---------------------------------------------------------------------|-------------------------|
| yolact_path           | Absolute path where *yolact* folder is located                      |           -             |
| model_path            | Absolute path where yolact weights are located (it should be located at yolact/weights/)      | - |
| image_topic           | Image topic used for subscribing                                    | /camera/color/image_raw |
| use_compressed_image  | Subscribe to compressed image topic                                 | False                   |
| publish_visualization | Publish images with detections                                      | True                    |
| publish_detections    | Publish detections as message                                       | True                    |
| display_visualization | Display window with detection image                                 | True                   |
| display_masks         | Whether or not to display masks over bounding boxes                 | True                    |
| display_bboxes        | Whether or not to display bboxes around masks                       | True                    |
| display_text          | Whether or not to display text (class [score])                      | True                    |
| display_scores        | Whether or not to display scores in addition to classes             | True                    |
| display_fps           | When displaying video, draw the FPS on the frame                    | True                   |
| score_threshold       | Detections with a score under this threshold will not be considered | 0.0                     |
| crop_masks            | If true, crop output masks with the predicted bounding box          | True                    |
| top_k                 | Further restrict the number of predictions to parse                 | 5                       |

If everything was okey, you should see something like this:

<p align="center">
  <img width="750" height="460" src="https://github.com/Eruvae/yolact_ros/tree/ROS2/docs/image_demo_yolact_ros2.png">
</p>

## Nodes

### yolact_ros2_node

*yolact_ros2_node* is located at *yolact_ros2/yolact_ros2.py*.

It takes as input the 2d image provided by camera driver and calculate 2d bounding boxes.

#### Topics

*yolact_ros2_node* publish information in two topics:

* **/yolact_ros2/detections:** In this topic, bounding box information is published. Here is published information of folloging type:

  **yolact_ros2_msgs/msg/Detections:**

  ```
  std_msgs/Header header
  Detection[] detections
  ```

  **yolact_ros2_msgs/msg/Detection:**

  ```
  string class_name
  float64 score
  Box box
  Mask mask
  ```

  where *class_name* is the name of the object detected, *score* is certainty probability, *box* contains bounding box limits in the following way:

  **yolact_ros2_msgs/msg/Box:**

  ```
  int32 x1
  int32 y1
  int32 x2
  int32 y2
  ```

  And *mask* provide which pixels contained in the bounding box belong to the detected object. It's format is as follow:

  **yolact_ros2_msgs/msg/Mask:**

  ```
  int32 width
  int32 height
  uint8[] mask # Mask encoded as bitset
  ```

  Where *width* and *height* is the size of bounding box and *mask* is an uint8 elements array whose size is (width*height) / 8 because each bit represents a boolean that indicates if that pixel takes part of the object or not.

* **/yolact_ros2/visualization:** In this topic is published messages of standard message type *sensor_msgs/Image* that you can visualize in *rviz*. It is useful for visual debugging:

<p align="center">
  <img width="750" height="460" src="https://github.com/Eruvae/yolact_ros/tree/ROS2/docs/image2_demo_yolact_ros2.png">
</p>
