# -*- coding: utf-8 -*-
# @Author: netrunner-exe
# @Date:   2022-12-21 12:52:01
# @Last Modified by:   netrunner-exe
# @Last Modified time: 2022-12-21 19:13:38
import logging

import tensorflow as tf
from tensorflow.keras.models import load_model
from networks.layers import InstanceNormalization

from networks.layers import AdaIN, AdaptiveAttention, AdaptiveAttentionSOA
from retinaface.models import *
from utils.options import FaceDancerOptions
from utils.swap_func import video_swap
from utils.hand_occlusion import create_hand_landmarker

logging.getLogger().setLevel(logging.ERROR)


if __name__ == '__main__':
    opt = FaceDancerOptions().parse()

    if len(tf.config.list_physical_devices('GPU')) != 0:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.set_visible_devices(gpus[opt.device_id], 'GPU')

    print('\nInitializing FaceDancer...')
    RetinaFace = load_model(opt.retina_path, compile=False,
                            custom_objects={"FPN": FPN,
                                            "SSH": SSH,
                                            "BboxHead": BboxHead,
                                            "LandmarkHead": LandmarkHead,
                                            "ClassHead": ClassHead})
    ArcFace = load_model(opt.arcface_path, compile=False)

    G = load_model(opt.facedancer_path, compile=False,
                   custom_objects={"AdaIN": AdaIN,
                                   "AdaptiveAttention": AdaptiveAttention,
                                   "AdaptiveAttentionSOA": AdaptiveAttentionSOA,
                                   "InstanceNormalization": InstanceNormalization})
    G.summary()

    print('\nProcessing: {}'.format(opt.vid_path))
    hand_landmarker = create_hand_landmarker(opt.hand_task_path)
    video_swap(opt, opt.swap_source, opt.vid_path,
               RetinaFace, ArcFace, G, opt.vid_output,
               hand_landmarker=hand_landmarker)
    hand_landmarker.close()
