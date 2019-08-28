#!/usr/bin/env python
import roslib
roslib.load_manifest('yolact_ros')
import sys
import rospy
import cv2
from std_msgs.msg import String
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from yolact_ros_msgs.msg import Detections
from yolact_ros_msgs.msg import Detection
from yolact_ros_msgs.msg import Box
from yolact_ros_msgs.msg import Mask
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import os
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from layers.output_utils import postprocess, undo_image_transformation
from data import COCODetection, get_label_map, MEANS, COLORS
from data import cfg, set_cfg, set_dataset
from utils import timer
from utils.functions import SavePath
from collections import defaultdict
from rospy.numpy_msg import numpy_msg

iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = {} # Call prep_coco_cats to fill this
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})

class image_converter:

  def __init__(self, net:Yolact):
    self.net = net

    self.image_pub = rospy.Publisher("cvimage_published",Image,queue_size=10)

    self.detections_pub = rospy.Publisher("detections",numpy_msg(Detections),queue_size=10)

    self.bridge = CvBridge()
    #self.image_sub = rospy.Subscriber("/sensorring_cam3d/rgb/image_raw",Image,self.callback)
    self.image_sub = rospy.Subscriber("/camera/rgb/image_raw",Image,self.callback)

  def prep_display(self, dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, image_header=Header()):
    with torch.no_grad():
        """
        Note: If undo_transform=False then im_h and im_w are allowed to be None.
        """
        dets = Detections()   

        if undo_transform:
            img_numpy = undo_image_transformation(img, w, h)
            img_gpu = torch.Tensor(img_numpy).cuda()
        else:
            img_gpu = img / 255.0
            h, w, _ = img.shape
        
        with timer.env('Postprocess'):
            t = postprocess(dets_out, w, h, visualize_lincomb = False,
                                            crop_masks        = True,
                                            score_threshold   = 0.3)
            torch.cuda.synchronize()

        with timer.env('Copy'):
            if cfg.eval_mask_branch:
                # Masks are drawn on the GPU, so don't copy
                masks = t[3][:100]
            classes, scores, boxes = [x[:100].cpu().numpy() for x in t[:3]]

        num_dets_to_consider = min(100, classes.shape[0])
        for j in range(num_dets_to_consider):
            if scores[j] < 0.3:
                num_dets_to_consider = j
                break
        
        if num_dets_to_consider == 0:
            # No detections found so just output the original image
            return (img_gpu * 255).byte().cpu().numpy()

        # Quick and dirty lambda for selecting the color for a particular index
        # Also keeps track of a per-gpu color cache for maximum speed
        def get_color(j, on_gpu=None):
            global color_cache
            color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)
            
            if on_gpu is not None and color_idx in color_cache[on_gpu]:
                return color_cache[on_gpu][color_idx]
            else:
                color = COLORS[color_idx]
                if not undo_transform:
                    # The image might come in as RGB or BRG, depending
                    color = (color[2], color[1], color[0])
                if on_gpu is not None:
                    color = torch.Tensor(color).to(on_gpu).float() / 255.
                    color_cache[on_gpu][color_idx] = color
                return color

        # First, draw the masks on the GPU where we can do it really fast
        # Beware: very fast but possibly unintelligible mask-drawing code ahead
        # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
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
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
            
        # Then draw the stuff that needs to be done on the cpu
        # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
        img_numpy = (img_gpu * 255).byte().cpu().numpy()
        
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j)
            score = scores[j]

            cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

            _class = cfg.dataset.class_names[classes[j]]
            text_str = '%s: %.2f' % (_class, score)

            font_face = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.6
            font_thickness = 1

            text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

            text_pt = (x1, y1 - 3)
            text_color = [255, 255, 255]

            cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
            cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
               
            det = Detection()
            det.box.x1 = x1
            det.box.y1 = y1
            det.box.x2 = x2
            det.box.y2 = y2
            det.class_name = _class
            det.score = score
            mask_shape = np.shape(masks[j])
            #print("Num dets: ",  num_dets_to_consider)
            #print("Shape: ", mask_shape)
            mask_bb = np.squeeze(masks[j].cpu().numpy(), axis=2)[y1:y2,x1:x2]
            #print("Box: ", x1,",",x2,",",y1,",",y2)
            #print("Mask in box shape: ", np.shape(mask_bb))
            mask_rs = np.reshape(mask_bb, -1)
            #print("New shape: ", np.shape(mask_rs))
            #print("Mask:\n",mask_bb)
            det.mask.height = y2 - y1
            det.mask.width = x2 - x1
            det.mask.mask = np.array(mask_rs, dtype=bool)
            dets.detections.append(det)
 
        dets.header.stamp = image_header.stamp
        dets.header.frame_id = image_header.frame_id
        self.detections_pub.publish(dets)
    return img_numpy

  def evalimage(self, cv_image, image_header):
    frame = torch.from_numpy(cv_image).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = self.net(batch)

    img_numpy = self.prep_display(preds, frame, None, None, undo_transform=False, image_header=image_header)
    
    #if save_path is None:
    #    img_numpy = img_numpy[:, :, (2, 1, 0)]

    cv2.imshow("Image window", img_numpy)
    cv2.waitKey(3)

  def callback(self, data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    self.evalimage(cv_image, data.header)

    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
      print(e)

def main(args):
  
  model_path = SavePath.from_str("weights/yolact_base_54_800000.pth")
  set_cfg(model_path.model_name + '_config')

  with torch.no_grad():
      if not os.path.exists('results'):
          os.makedirs('results')

      cudnn.benchmark = True
      cudnn.fastest = True
      torch.set_default_tensor_type('torch.cuda.FloatTensor')   

      print('Loading model...', end='')
      net = Yolact()
      net.load_weights("weights/yolact_base_54_800000.pth")
      net.eval()
      print(' Done.')

      net = net.cuda()
      net.detect.use_fast_nms = True
      cfg.mask_proto_debug = False


  ic = image_converter(net)
  rospy.init_node('image_converter', anonymous=True)

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
