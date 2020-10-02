#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.exceptions import ParameterNotDeclaredException
from rclpy.duration import Duration
from rcl_interfaces.msg import SetParametersResult
from rcl_interfaces.msg import Parameter, ParameterType

import sys
import os
import cv2
import threading
from queue import Queue
from std_msgs.msg import String
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from yolact_ros2_msgs.msg import Detections
from yolact_ros2_msgs.msg import Detection
from yolact_ros2_msgs.msg import Box
from yolact_ros2_msgs.msg import Mask
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from collections import defaultdict

iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = {}
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})

class YolactNode(Node):
    def __init__(self, node_name):
        super().__init__(node_name)
        self.initParams_()
        self.model_path = None
        self.image_sub = None
        self.received_img = None
        
        self.image_vis_queue = Queue(maxsize = 1)
        self.visualization_thread = None
        self.unpause_visualization = threading.Event()
        self.bridge = CvBridge()

        self.declare_parameters(
        namespace='',
        parameters=[
            ('yolact_path', self.yolact_path_),
            ('model_path', self.model_path_),
            ('image_topic', self.image_topic_),
            ('use_compressed_image', self.use_compressed_image_),
            ('publish_visualization', self.publish_visualization_),
            ('publish_detections', self.publish_detections_),
            ('display_visualization', self.display_visualization_),
            ('display_masks', self.display_masks_),
            ('display_bboxes', self.display_bboxes_),
            ('display_text', self.display_text_),
            ('display_scores', self.display_scores_),
            ('display_fps', self.display_fps_),
            ('score_threshold', self.score_threshold_),
            ('crop_masks', self.crop_masks_),
            ('top_k', self.top_k_),
            ('publish_namespace', '/yolact_ros2')
        ])
        publish_ns = self.get_parameter('publish_namespace')._value
        self.image_pub = self.create_publisher(Image, f'{publish_ns}/visualization', 1)
        self.detections_pub = self.create_publisher(Detections, f'{publish_ns}/detections', 1)
        self.image_pub = self.create_publisher(Image, '/yolact_ros2/visualization', 1)
        self.detections_pub = self.create_publisher(Detections, '/yolact_ros2/detections', 1)
        self.setParams_()

        # Set Reconfigurable parameters Callback:

        self.set_parameters_callback(self.parameter_callback_)

        sys.path.append(self.yolact_path_)

        self.loadWeights_()

        self.set_subscription_()

        #for counting fps
        self.fps = 0
        self.last_reset_time = self.get_clock().now()
        self.frame_counter = 0

    def loadWeights_(self):
        from yolact import Yolact
        from layers.output_utils import undo_image_transformation
        from data import COCODetection, get_label_map, MEANS
        from data import cfg, set_cfg, set_dataset
        from utils.functions import SavePath
        try:
            self.model_path = SavePath.from_str(self.model_path_)
        except ValueError:
            self.get_logger().error("File [" + self.model_path_ + "] is not correct format")
            sys.exit(1)

        set_cfg(self.model_path.model_name + '_config')

        with torch.no_grad():
            cudnn.benchmark = True
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

            self.get_logger().info('Loading model from ' + self.model_path_)
            self.net = Yolact()
            try:
                self.net.load_weights(self.model_path_)
            except FileNotFoundError:
                self.get_logger().error("File [" + self.model_path_ + "] doesn't exist")
                sys.exit(1)
            self.net.eval()
            self.get_logger().info('Done.')

            self.net = self.net.cuda()
            self.net.detect.use_fast_nms = True
            cfg.mask_proto_debug = False

    def initParams_(self):
        self.yolact_path_ = None
        self.model_path_ = None
        self.image_topic_ = '/camera/color/image_raw'
        self.use_compressed_image_ = False
        self.publish_visualization_ = True
        self.publish_detections_ = True
        self.display_visualization_ = False
        self.display_masks_ = True
        self.display_bboxes_ = True
        self.display_text_ = True
        self.display_scores_ = True
        self.display_fps_ = False
        self.score_threshold_ = 0.0
        self.crop_masks_ = True
        self.top_k_ = True

    def setParams_(self):
        self.yolact_path_ = self.get_parameter('yolact_path')._value
        self.model_path_ = self.get_parameter('model_path')._value
        self.image_topic_ = self.get_parameter('image_topic')._value
        self.use_compressed_image_ = self.get_parameter('use_compressed_image')._value
        self.publish_visualization_ = self.get_parameter('publish_visualization')._value
        self.publish_detections_ = self.get_parameter('publish_detections')._value
        self.display_visualization_ = self.get_parameter('display_visualization')._value
        self.display_masks_ = self.get_parameter('display_masks')._value
        self.display_bboxes_ = self.get_parameter('display_bboxes')._value
        self.display_text_ = self.get_parameter('display_text')._value
        self.display_scores_ = self.get_parameter('display_scores')._value
        self.display_fps_ = self.get_parameter('display_fps')._value
        self.score_threshold_ = self.get_parameter('score_threshold')._value
        self.crop_masks_ = self.get_parameter('crop_masks')._value
        self.top_k_ = self.get_parameter('top_k')._value

    def parameter_callback_(self, params):
        model_path_changed = False
        for param in params:
            if (param.name == 'yolact_path' and param.type_ == param.Type.STRING):
                self.yolact_path_ = param.value
                continue
            if (param.name == 'model_path' and param.type_ == param.Type.STRING):
                self.model_path_ = param.value
                model_path_changed = True
                continue
            if (param.name == 'image_topic' and param.type_ == param.Type.STRING):
                self.image_topic_ = param.value
                continue
            if (param.name == 'use_compressed_image' and param.type_ == param.Type.BOOL):
                self.use_compressed_image_ = param.value
                continue
            if (param.name == 'publish_visualization' and param.type_ == param.Type.BOOL):
                self.publish_visualization_ = param.value
                continue
            if (param.name == 'publish_detections' and param.type_ == param.Type.BOOL):
                self.publish_detections_ = param.value
                continue
            if (param.name == 'display_visualization' and param.type_ == param.Type.BOOL):
                self.display_visualization_ = param.value
                continue
            if (param.name == 'display_masks' and param.type_ == param.Type.BOOL):
                self.display_masks_ = param.value
                continue
            if (param.name == 'display_bboxes' and param.type_ == param.Type.BOOL):
                self.display_bboxes_ = param.value
                continue
            if (param.name == 'display_text' and param.type_ == param.Type.BOOL):
                self.display_text_ = param.value
                continue
            if (param.name == 'display_scores' and param.type_ == param.Type.BOOL):
                self.display_scores_ = param.value
                continue
            if (param.name == 'display_fps' and param.type_ == param.Type.BOOL):
                self.display_fps_ = param.value
                continue
            if (param.name == 'score_threshold' and param.type_ == param.Type.DOUBLE):
                self.score_threshold_ = param.value
                continue
            if (param.name == 'crop_masks' and param.type_ == param.Type.BOOL):
                self.crop_masks_ = param.value
                continue
            if (param.name == 'top_k' and param.type_ == param.Type.BOOL):
                self.top_k_ = param.value

        self.get_logger().warn('****PARAMETERS CHANGED****')

        if (model_path_changed):
            self.loadWeights_()

        return SetParametersResult(successful=True)

    def set_subscription_(self):
        if (self.use_compressed_image_):
            self.create_subscription(CompressedImage, '/compressed', self.img_callback_, 1)
        else:
            self.create_subscription(Image, self.image_topic_, self.img_callback_, 1)

        if (self.display_visualization_):
            self.unpause_visualization.set()
            if self.visualization_thread is None: # first time visualization
                self.get_logger().info('Creating thread')
                self.visualization_thread = threading.Thread(target=self.visualizationLoop_)
                self.visualization_thread.daemon = True
                self.visualization_thread.start()
                self.get_logger().info('Thread was started')
        else:
            self.unpause_visualization.clear()

    def img_callback_(self, msg):

        try:
            if (self.use_compressed_image_):
                np_arr = np.fromstring(msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(e)

        self.evalimage_(cv_image, msg.header)

    def evalimage_(self, cv_image, image_header):
        from utils.augmentations import BaseTransform, FastBaseTransform, Resize
        with torch.no_grad():
            frame = torch.from_numpy(cv_image).cuda().float()
            batch = FastBaseTransform()(frame.unsqueeze(0))
            preds = self.net(batch)

            h, w, _ = frame.shape
            classes, scores, boxes, masks = self.postprocess_results_(preds, w, h)

            if (self.display_fps_):
                now = self.get_clock().now()
                if now - self.last_reset_time > Duration(seconds=1): # reset timer / counter every second
                    self.fps = self.frame_counter
                    self.last_reset_time = now
                    self.frame_counter = 0
                self.frame_counter += 1

            if (self.publish_visualization_ or self.display_visualization_):
                image = self.prep_display_(classes, scores, boxes, masks, frame, fps_str=str(self.fps))

            if (self.publish_detections_):
                dets = self.generate_detections_msg_(classes, scores, boxes, masks, image_header)
                self.detections_pub.publish(dets)

            if (self.display_visualization_ and not self.image_vis_queue.full()):
                self.image_vis_queue.put_nowait(image)

            if (self.publish_visualization_):
                try:
                    image = self.bridge.cv2_to_imgmsg(image, "bgr8")
                    image.header = image_header
                    self.image_pub.publish(image)
                except CvBridgeError as e:
                    print(e)

    def postprocess_results_(self, dets_out, w, h):
        from utils import timer
        from data import cfg
        from layers.output_utils import postprocess
        with timer.env('Postprocess'):
            save = cfg.rescore_bbox
            cfg.rescore_bbox = True
            t = postprocess(dets_out, w, h, visualize_lincomb = False,
                                            crop_masks        = self.crop_masks_,
                                            score_threshold   = self.score_threshold_)
            cfg.rescore_bbox = save

        with timer.env('Copy'):
            idx = t[1].argsort(0, descending=True)[:self.top_k_]

            if (cfg.eval_mask_branch):
                # Masks are drawn on the GPU, so don't copy
                masks = t[3][idx]
            classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

        return classes, scores, boxes, masks

    def prep_display_(self, classes, scores, boxes, masks, img, class_color=False, mask_alpha=0.45, fps_str=''):
        from data import cfg, COLORS
        img_gpu = img / 255.0

        num_dets_to_consider = min(self.top_k_, classes.shape[0])
        for j in range(num_dets_to_consider):
            if (scores[j] < self.score_threshold_):
                num_dets_to_consider = j
                break

        # Quick and dirty lambda for selecting the color for a particular index
        # Also keeps track of a per-gpu color cache for maximum speed
        def get_color(j, on_gpu=None):
            global color_cache
            color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)

            if on_gpu is not None and color_idx in color_cache[on_gpu]:
                return color_cache[on_gpu][color_idx]
            else:
                color = COLORS[color_idx]
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
                if on_gpu is not None:
                    color = torch.Tensor(color).to(on_gpu).float() / 255.
                    color_cache[on_gpu][color_idx] = color
                return color

        # First, draw the masks on the GPU where we can do it really fast
        # Beware: very fast but possibly unintelligible mask-drawing code ahead
        # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
        if (self.display_masks_ and cfg.eval_mask_branch and num_dets_to_consider > 0):
            # After this, mask is of size [num_dets, h, w, 1]
            masks = masks[:num_dets_to_consider, :, :, None]

            # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
            colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
            masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

            # This is 1 everywhere except for 1-mask_alpha where the mask is
            inv_alph_masks = masks * (-mask_alpha) + 1

            # I did the math for this on pen and paper. This whole block should be equivalent to:
            #    for j in range(num_dets_to_consider):
            #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
            masks_color_summand = masks_color[0]
            if (num_dets_to_consider > 1):
                inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
                masks_color_cumul = masks_color[1:] * inv_alph_cumul
                masks_color_summand += masks_color_cumul.sum(dim=0)

            img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

        if (self.display_fps_):
            # Draw the box for the fps on the GPU
            font_face = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.6
            font_thickness = 1

            text_w, text_h = cv2.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]

            img_gpu[0:text_h+8, 0:text_w+8] *= 0.6 # 1 - Box alpha


        # Then draw the stuff that needs to be done on the cpu
        # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
        img_numpy = (img_gpu * 255).byte().cpu().numpy()

        if (self.display_fps_):
            # Draw the text on the CPU
            text_pt = (4, text_h + 2)
            text_color = [255, 255, 255]

            cv2.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

        if num_dets_to_consider == 0:
            return img_numpy

        if (self.display_text_ or self.display_bboxes_):
            for j in reversed(range(num_dets_to_consider)):
                x1, y1, x2, y2 = boxes[j, :]
                color = get_color(j)
                score = scores[j]

                if (self.display_bboxes_):
                    cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

                if (self.display_text_):
                    _class = cfg.dataset.class_names[classes[j]]
                    text_str = '%s: %.2f' % (_class, score) if self.display_scores_ else _class

                    font_face = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 0.6
                    font_thickness = 1

                    text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                    text_pt = (x1, y1 - 3)
                    text_color = [255, 255, 255]

                    cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                    cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)


        return img_numpy

    def generate_detections_msg_(self, classes, scores, boxes, masks, image_header):
        from data import cfg
        dets_msg = Detections()
        for detnum in range(len(classes)):
            det = Detection()
            det.class_name = cfg.dataset.class_names[classes[detnum]]
            det.score = float(scores[detnum])
            x1, y1, x2, y2 = boxes[detnum]
            det.box.x1 = int(x1)
            det.box.y1 = int(y1)
            det.box.x2 = int(x2)
            det.box.y2 = int(y2)
            mask = masks[detnum,y1:y2,x1:x2]
            det.mask.mask = np.packbits(mask.bool().cpu()).tolist()
            det.mask.height = int(y2 - y1)
            det.mask.width = int(x2 - x1)
            dets_msg.detections.append(det)

        dets_msg.header = image_header
        return dets_msg

    def visualizationLoop_(self):
        print('Creating cv2 window')
        window_name = 'Segmentation results'
        cv2.namedWindow(window_name)
        self.get_logger().info('Window successfully created')
        while True:
          if (not self.unpause_visualization.is_set()):
              self.get_logger().info('Pausing visualization')
              cv2.destroyWindow(window_name)
              cv2.waitKey(30)
              self.unpause_visualization.wait()
              self.get_logger().info('Unpausing visualization')
              cv2.namedWindow(window_name)

          if (self.image_vis_queue.empty()):
              cv2.waitKey(30)
              continue

          image = self.image_vis_queue.get_nowait()
          cv2.imshow(window_name, image)
          cv2.waitKey(30)

def main(args=None):
    rclpy.init(args=args)

    # Create node:

    yolact_node = YolactNode('yolact_ros2_node')

    try:
        rclpy.spin(yolact_node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()

if __name__ == '__main__':
    main(sys.argv)
