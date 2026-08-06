# -*- coding: utf-8 -*-
# @Author: netrunner-exe
# @Date:   2022-12-21 12:52:01
# @Last Modified by:   netrunner-exe
# @Last Modified time: 2022-12-21 19:14:34
import logging
import os
from pathlib import Path

os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')

import tensorflow as tf
from tensorflow.keras.models import load_model
from networks.layers import InstanceNormalization

from networks.layers import AdaIN, AdaptiveAttention, AdaptiveAttentionSOA
from retinaface.models import *
from utils.options import FaceDancerOptions
from utils.swap_func import run_inference
from utils.hand_occlusion import create_hand_landmarker
from utils.utils import load_model_internal

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

    facedancer_path = Path(opt.facedancer_path)
    checkpoint_prefix = 'gen_'
    is_internal_checkpoint = (
        facedancer_path.suffix.lower() == '.h5'
        and facedancer_path.stem.startswith(checkpoint_prefix)
        and (facedancer_path.parent / 'gen.json').is_file()
    )
    if is_internal_checkpoint:
        checkpoint_number = facedancer_path.stem[len(checkpoint_prefix):]
        G = load_model_internal(
            str(facedancer_path.parent) + os.sep,
            'gen', checkpoint_number
        )
    else:
        G = load_model(opt.facedancer_path, compile=False,
                       custom_objects={"AdaIN": AdaIN,
                                       "AdaptiveAttention": AdaptiveAttention,
                                       "AdaptiveAttentionSOA": AdaptiveAttentionSOA,
                                       "InstanceNormalization": InstanceNormalization})
    G.summary()

    hand_landmarker = create_hand_landmarker(opt.hand_task_path)
    Path(opt.img_output).parent.mkdir(parents=True, exist_ok=True)
    print('\nProcessing: {}'.format(opt.img_path))
    run_inference(opt, opt.swap_source, opt.img_path,
                  RetinaFace, ArcFace, G, opt.img_output,
                  hand_landmarker=hand_landmarker,
                  debug=opt.debug_hand_mask)
    hand_landmarker.close()
    print('\nDone! {}'.format(opt.img_output))
