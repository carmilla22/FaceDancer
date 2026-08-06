import logging
import tensorflow as tf
from tensorflow.keras.models import load_model
from networks.layers import InstanceNormalization
from networks.layers import AdaIN, AdaptiveAttention, AdaptiveAttentionSOA
from retinaface.models import *
from utils.options import FaceDancerOptions
from utils.swap_func import run_inference
from utils.hand_occlusion import create_hand_landmarker
from fastapi import FastAPI, File, UploadFile, Form
import uvicorn
import os, shutil
import threading
from pathlib import Path
from fastapi import HTTPException
from fastapi.responses import FileResponse

logging.getLogger().setLevel(logging.ERROR)

opt = FaceDancerOptions().parse()

if len(tf.config.list_physical_devices('GPU')) != 0:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[opt.device_id], 'GPU')

print('\nInitializing FaceDancer...')
RetinaFace = load_model(opt.retina_path, compile=False, custom_objects = {"FPN": FPN,
                                                                          "SSH": SSH,
                                                                          "BboxHead": BboxHead,
                                                                          "LandmarkHead": LandmarkHead,
                                                                          "ClassHead": ClassHead})

ArcFace = load_model(opt.arcface_path, compile=False)

G = load_model(opt.facedancer_path, compile=False, custom_objects={"AdaIN": AdaIN,
                                                                   "AdaptiveAttention": AdaptiveAttention,
                                                                   "AdaptiveAttentionSOA": AdaptiveAttentionSOA,
                                                                   "InstanceNormalization": InstanceNormalization})
if len(G.inputs) != 3:
    raise RuntimeError(
        'The configured generator is not FaceDancer-SOA. '
        'Train/export a model with three inputs or pass --facedancer_path '
        'to a FaceDancer-SOA .h5 file.'
    )
G.summary()

hand_landmarker = None
try:
    hand_landmarker = create_hand_landmarker(opt.hand_task_path)
    print('Hand Landmarker initialized.')
except Exception as error:
    print('WARN: failed to initialize Hand Landmarker: {}'.format(error))

# TensorFlow and MediaPipe inference are serialized for this single-process API.
inference_lock = threading.Lock()


app = FastAPI()


@app.on_event("shutdown")
def close_models():
    if hand_landmarker is not None:
        hand_landmarker.close()
    
@app.get("/")
def test_api():
    return {"Hello": "World"}

@app.post("/faceswap")
def generate(content_image: UploadFile = File(...), target_image: UploadFile = File(...)):
    os.makedirs('./results', exist_ok=True)
    os.makedirs('./source_image', exist_ok=True)
    os.makedirs('./swap_image', exist_ok=True)
    
    # Never use an upload-provided directory component in a local path.
    source_name = Path(content_image.filename or 'source.png').name
    target_name = Path(target_image.filename or 'target.png').name

    input_image_file_path1 = os.path.join('./source_image', source_name)
    with open(input_image_file_path1, "wb") as buffer:
        shutil.copyfileobj(content_image.file, buffer)
    
    input_image_file_path2 = os.path.join('./swap_image', target_name)
    with open(input_image_file_path2, "wb") as buffer:
        shutil.copyfileobj(target_image.file, buffer)

    print('\nProcessing: {}'.format(input_image_file_path2))
    try:
        with inference_lock:
            run_inference(
                opt, input_image_file_path1, input_image_file_path2,
                RetinaFace, ArcFace, G, opt.img_output,
                hand_landmarker=hand_landmarker,
                debug=opt.debug_hand_mask
            )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    print('\nDone! {}'.format(opt.img_output))
    return FileResponse(opt.img_output, media_type='image/jpeg')


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
