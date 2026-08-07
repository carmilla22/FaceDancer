<div align="center">

# <b>FaceDancer: Pose- and Occlusion-Aware High Fidelity Face Swapping</b>
![demo_vid_0](assets/133_to_4.gif)

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2210.10473) [![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)  [![GitHub Stars](https://img.shields.io/github/stars/felixrosberg/FaceDancer?affiliations=OWNER&color=green&style=social)](https://github.com/felixrosberg/FaceDancer) ![visitors](https://visitor-badge.laobi.icu/badge?page_id=felixrosberg/FaceDancer) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/felixrosberg/face-swap) <a href="https://colab.research.google.com/github/felixrosberg/FaceDancer/blob/main/FaceDancer_colab_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>

[[**WACV 2023**](https://openaccess.thecvf.com/content/WACV2023/html/Rosberg_FaceDancer_Pose-_and_Occlusion-Aware_High_Fidelity_Face_Swapping_WACV_2023_paper.html)] [[**Video Results**](https://drive.google.com/drive/folders/1hHjK0W-Oo1HD6OZb97IdSifPs4_c6NNo?usp=sharing)]

</div>


#### 🚩 Updates:
#### **06.01.2023:**


- Dockerfile and API examples provided by [AliYqb](https://github.com/AliYqb).
#### **03.01.2023:**


- Minor code improvements. Fixed ```No module named 'retinaface.models'``` error (thanks to [kambiannan](https://github.com/kambiannan)).
- Added ability to use a GIF as an input file and export the final result to GIF or WEBP. For more information, check the updated [installation guide](https://github.com/felixrosberg/FaceDancer#installation).


## Abstract
>In this work, we present a new single-stage method for
>subject agnostic face swapping and identity transfer, named
>FaceDancer. We have two major contributions: Adaptive
>Feature Fusion Attention (AFFA) and Interpreted Feature
>Similarity Regularization (IFSR). The AFFA module is embedded
> in the decoder and adaptively learns to fuse attribute
> features and features conditioned on identity information
> without requiring any additional facial segmentation process.
>In IFSR, we leverage the intermediate features
> in an identity encoder to preserve important attributes
> such as head pose, facial expression, lighting, and occlusion
> in the target face, while still transferring the identity
> of the source face with high fidelity. We conduct extensive
> quantitative and qualitative experiments on various
> datasets and show that the proposed FaceDancer outperforms
> other state-of-the-art networks in terms of identity
> transfer, while having significantly better pose preservation
> than most of the previous methods.

![overview](assets/facedancer_ov.png)

For a quick play around, you can check out a version of FaceDancer hosted on [Hugging Face](https://huggingface.co/spaces/felixrosberg/face-swap). The Space allow you to face swap images, but also try some other functionality I am currently researching, which I plan to publish soon. For example, reconstruction attacks and adversarial defense against the reconstruction attacks.

![result_matrix](assets/result_matrix.png)

## Getting Started
This project was implemented in TensorFlow 2.X. For evaluation we used models implemented in both TensorFlow and PyTorch (e.g CosFace from [InsightFace](https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch)).

### Installation:
#### Here is an example of installing FaceDancer on Windows:

- Clone or download repository
```shell
git clone https://github.com/felixrosberg/FaceDancer.git
cd FaceDancer
```

- Make conda environment
```shell
conda create -n facedancer python=3.8
conda activate facedancer
python -m pip install --upgrade pip
```

- Download and install [Microsoft Visual C++ for Visual Studio 2015](https://support.microsoft.com/help/2977003/the-latest-supported-visual-c-downloads?utm_source=www.tensorflow.org&utm_medium=referral) (if you not installed it)

- The easiest way to run FaceDancer on GPU is to install **tensorflow-cpu** and **tensorflow-directml-plugin**. if you need only a CPU, then the installation of **tensorflow-directml-plugin** can be skipped. To work with the GPU, the latest Nvidia driver must be installed.

```shell
conda install -c conda-forge ffmpeg

pip install tensorflow-cpu==2.10
pip install tensorflow-directml-plugin
```

- Install depencies:
```shell
pip install -r requirements.txt
```


#### An alternative installation method if you have difficulty with the previous:
- Clone or download repository
```shell
git clone https://github.com/felixrosberg/FaceDancer.git
cd FaceDancer
```

- Make conda environment
```shell
conda create -n facedancer python=3.8
conda activate facedancer
python -m pip install --upgrade pip
```

- Install depencies:
```shell
conda install -c conda-forge cudatoolkit cudnn
conda install -c conda-forge ffmpeg

pip install tensorflow-gpu
pip install -r requirements.txt
```

#### Models:
Download the pretrained ArcFace **[here](https://huggingface.co/felixrosberg/ArcFace)** (only **ArcFace-Res50.h5** is needed for swapping) and RetinaFace **[here](https://huggingface.co/felixrosberg/RetinaFace)**. Secondly you need to train FaceDancer or **download a pretrained model weights from [here](https://huggingface.co/felixrosberg/FaceDancer)**.
- Put **ArcFace-Res50.h5** inside the **./arcface_model** dir.
- Put **RetinaFace-Res50.h5** inside the **./retinaface** dir.
- Put downloaded pretrained models inside the **./model_zoo** dir.

#
### To swap all faces with one source, run:
> **Warning**
>#### *Source image with too high resolution may not work properly!*
#### Video:
```shell
python test_video_swap_multi.py --facedancer_path "./model_zoo/FaceDancer_config_c_HQ.h5" --vid_path "path/to/video.mp4" --swap_source "path/to/source_face.jpg" --vid_output "results/swapped_video.mp4"
```

#### Image:
```shell
python test_image_swap_multi.py --facedancer_path "./model_zoo/FaceDancer_config_c_HQ.h5" --img_path "path/to/image.jpg" --swap_source "path/to/source_face.jpg" --img_output "results/swapped_image.jpg"
```


#### GIF:
```shell
python test_video_swap_multi.py --facedancer_path "./model_zoo/FaceDancer_config_c_HQ.h5" --vid_path "path/to/GIF_or_MP4.gif" --swap_source "path/to/source_face.jpg" --vid_output "results/swapped_GIF_or_WEBP.gif"
```


> **Note**
>
> *You can use video or GIF as input file, so that the final file will be in GIF or WEBP format, just change the extension in **--vid_output** to **.gif** or **.webp**.*


#### *The video, GIF or image with swapped faces will be saved in the **./results** directory*


## Using the Models in Custom script
```python
import logging

import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from networks.layers import AdaIN, AdaptiveAttention, InstanceNormalization

logging.getLogger().setLevel(logging.ERROR)


model = load_model("path/to/model.h5", compile=False, custom_objects={"AdaIN": AdaIN,
                                                                      "AdaptiveAttention": AdaptiveAttention,
                                                                      "InstanceNormalization": InstanceNormalization})
arcface = load_model("path/to/arcface.h5", compile=False)

# target and source images need to be properly cropeed and aligned
target = np.asarray(Image.open("path/to/target_face.png").resize((256, 256)))
source = np.asarray(Image.open("path/to/source_face.png").resize((112, 112)))

source_z = arcface(np.expand_dims(source / 255.0, axis=0))

face_swap = model([np.expand_dims((target - 127.5) / 127.5, axis=0), source_z]).numpy()
face_swap = (face_swap[0] + 1) / 2
face_swap = np.clip(face_swap * 255, 0, 255).astype('uint8')

cv2.imwrite("./swapped_face.png", cv2.cvtColor(face_swap, cv2.COLOR_BGR2RGB))
```


> **Note**
>
> *The important part is that you need ArcFace as well and make sure the target image is normalized between -1 and 1, and the source between 0 and 1.*


## How to Preprocess Data

### Aligning Faces
Before you can train FaceDancer you must make sure the data is properly aligned and processed. Learning capabilites is crippled without this step, if not impossible. The expected folder structure is DATASET/subfolders/im_0, ..., im_x. If using an image dataset not divided into subfolders you can put the DATASET folder inside a parent folder like this: PARENT_FOLDER/DATASET/im_0, ..., im_x. Then specify the PARENT_FOLDER as the --data_dir and the DATASET will be treated as a subfolder. This step requires the pretrained RetinaFace for face detection and facial landmark extraction.

To align the faces run:
```shell
python dataset/crop_align.py --data_dir path/to/DATASET --target_dir path/to/processed_DATASET
```

Remaining arguments consist of:
- **--device_id, default=0** - *Which device to use.*
- **--im_size, default=256** - *Final image size of the processed image.*
- **--min_size, default=128** - *Threshold to ignore image with a width or height lower than min_size.*
- **--shrink_factor, default=1.0** - *This argument controls how much of the background to keep. Default is 1.0 which produces images appropriate as direct input into ArcFace. If the shrink factor is e.g 0.75, you must center crop the image, keeping 75% of the image, before inputting into ArcFace.*

### Sharding the Data
This step will convert the image data to tfrecords. If using large datasets such as VGGFace2 this will take some time. However, the training code is designed around this step and it speeds up training significantly. The expected folder structure is DATASET/subfolders/im_0, ..., im_x. If using an image dataset not divided into subfolders you can put the DATASET folder inside a parent folder like this: PARENT_FOLDER/DATASET/im_0, ..., im_x. Then specify the PARENT_FOLDER as the --data_dir and the DATASET will be treated as a subfolder.

Training uses four independently shuffled TFRecord sets: target training,
target validation, hand-occluded source training, and hand-occluded source
validation. The following commands write the complete datasets with 1,000
images per shard. `--max_images` is intentionally omitted.

```shell
devenv shell -- python dataset/dataset_sharding.py \
  --data_dir /srv/facedancer/images/target/train \
  --target_dir /srv/facedancer/tfrecords/target/train \
  --data_name target --data_type train --num_shards 1000

devenv shell -- python dataset/dataset_sharding.py \
  --data_dir /srv/facedancer/images/target/validation \
  --target_dir /srv/facedancer/tfrecords/target/validation \
  --data_name target --data_type validation --num_shards 1000

devenv shell -- python dataset/dataset_sharding.py \
  --data_dir /srv/facedancer/images/source-occluded/train \
  --target_dir /srv/facedancer/tfrecords/source-occluded/train \
  --data_name source_occluded --data_type train --num_shards 1000

devenv shell -- python dataset/dataset_sharding.py \
  --data_dir /srv/facedancer/images/source-occluded/validation \
  --target_dir /srv/facedancer/tfrecords/source-occluded/validation \
  --data_name source_occluded --data_type validation --num_shards 1000
```

Remaining arguments consist of:
- **--data_type, default="train"** - *Identifier for the output file names.*
- **--shuffle, default=True** - *Where to shuffle the order of sharding the images.*
- **--num_shards, default=1000** - *Number of images written to each shard (the argument name is historical).*
- **--max_images, default=None** - *Optional image limit for smoke-test shards; omit it for full datasets.*

## How to Train
After processing and sharding the datasets, download the pretrained ArcFace
model **[here](https://huggingface.co/felixrosberg/ArcFace)** and the expression
embedding model **[here](https://huggingface.co/felixrosberg/ExpressionEmbedder)**.
The commands below assume they are stored under `/srv/facedancer/models` and
that this repository is the current directory. Replace `/srv/facedancer` with
the persistent absolute data path on the training host.

Validate the frozen native environment and a real TensorFlow GPU operation
before training:

```shell
devenv shell -- fd-check-env
```

The command must report the NVIDIA GPU and a successful GPU matrix
multiplication. On a host without an NVIDIA device, imports can be checked but
GPU acceptance cannot be completed.

To split an image dataset before sharding it, run:
```shell
devenv shell -- python dataset/split_dataset.py \
  --input_dir /srv/facedancer/images/all \
  --output_dir /srv/facedancer/images/split \
  --train_fraction 0.8 --seed 42
```
The command copies the images into `train` and `validation` while preserving
the source dataset and any nested folder structure.

The hand-occluded source images must be sharded into separate training and
validation TFRecords, just like the target dataset. Hand masks are not stored
in the dataset: MediaPipe detects them online during training and validation.
The source and target streams are shuffled and repeated independently; they do
not need identity pairing.

Training batches use `drop_remainder=True`. Configure one exhaustive epoch as:

```text
iterations_per_epoch = training_images // batch_size
```

For 30,000 images per training stream, use 30,000 iterations at batch size 1
or 15,000 iterations at batch size 2. A non-divisible final partial batch is
not consumed. Start with batch size 1 on a 24 GB GPU and only increase it after
measuring peak memory use.

### Checkpoint and Resume Acceptance

First run one 10-step epoch. This saves completed-iteration checkpoint `10`:

```shell
devenv shell -- python train/train.py \
  --data_dir '/srv/facedancer/tfrecords/target/train/target_train_*-of-*.records' \
  --source_data_dir '/srv/facedancer/tfrecords/source-occluded/train/source_occluded_train_*-of-*.records' \
  --eval_dir '/srv/facedancer/tfrecords/target/validation/target_validation_*-of-*.records' \
  --eval_source_dir '/srv/facedancer/tfrecords/source-occluded/validation/source_occluded_validation_*-of-*.records' \
  --arcface_path /srv/facedancer/models/ArcFace-Res50.h5 \
  --eval_model_expface /srv/facedancer/models/ExpressionEmbedder-B0.h5 \
  --hand_task_path /srv/facedancer/models/hand_landmarker.task \
  --batch_size 1 --eval_batch_size 1 \
  --iterations_per_epoch 10 --num_epochs 1 \
  --checkpoint_interval 10 --device_id 0 \
  --log_dir /srv/facedancer/logs \
  --chkp_dir /srv/facedancer/checkpoints \
  --log_name soa-checkpoint-acceptance
```

Then resume the same run with a total target of two epochs:

```shell
devenv shell -- python train/train.py \
  --data_dir '/srv/facedancer/tfrecords/target/train/target_train_*-of-*.records' \
  --source_data_dir '/srv/facedancer/tfrecords/source-occluded/train/source_occluded_train_*-of-*.records' \
  --eval_dir '/srv/facedancer/tfrecords/target/validation/target_validation_*-of-*.records' \
  --eval_source_dir '/srv/facedancer/tfrecords/source-occluded/validation/source_occluded_validation_*-of-*.records' \
  --arcface_path /srv/facedancer/models/ArcFace-Res50.h5 \
  --eval_model_expface /srv/facedancer/models/ExpressionEmbedder-B0.h5 \
  --hand_task_path /srv/facedancer/models/hand_landmarker.task \
  --batch_size 1 --eval_batch_size 1 \
  --iterations_per_epoch 10 --num_epochs 2 \
  --checkpoint_interval 10 --device_id 0 \
  --log_dir /srv/facedancer/logs \
  --chkp_dir /srv/facedancer/checkpoints \
  --log_name soa-checkpoint-acceptance \
  --load 10
```

The resumed run starts at epoch 2, epoch iteration 1, global completed
iteration 10. It performs exactly 10 more steps and creates:

```text
/srv/facedancer/checkpoints/soa-checkpoint-acceptance/gen/gen_20.h5
/srv/facedancer/checkpoints/soa-checkpoint-acceptance/dis/dis_20.h5
/srv/facedancer/checkpoints/soa-checkpoint-acceptance/state/20.json
```

### Full-Dataset GPU Validation

After checkpoint/resume acceptance, run one complete shuffled 30,000-image
pass before starting a multi-epoch production session:

```shell
devenv shell -- python train/train.py \
  --data_dir '/srv/facedancer/tfrecords/target/train/target_train_*-of-*.records' \
  --source_data_dir '/srv/facedancer/tfrecords/source-occluded/train/source_occluded_train_*-of-*.records' \
  --eval_dir '/srv/facedancer/tfrecords/target/validation/target_validation_*-of-*.records' \
  --eval_source_dir '/srv/facedancer/tfrecords/source-occluded/validation/source_occluded_validation_*-of-*.records' \
  --arcface_path /srv/facedancer/models/ArcFace-Res50.h5 \
  --eval_model_expface /srv/facedancer/models/ExpressionEmbedder-B0.h5 \
  --hand_task_path /srv/facedancer/models/hand_landmarker.task \
  --batch_size 1 --eval_batch_size 1 \
  --iterations_per_epoch 30000 --num_epochs 1 \
  --checkpoint_interval 10000 --device_id 0 \
  --log_dir /srv/facedancer/logs \
  --chkp_dir /srv/facedancer/checkpoints \
  --log_name soa-gpu-validation
```

This creates checkpoints `10000`, `20000`, and `30000`; the final checkpoint
at the interval boundary is not written twice.

### Configurable Production Run

Set the image count, batch size, and operator-chosen total epoch target. If the
batch size changes for a fresh run, recalculate `STEPS_PER_EPOCH`.

```shell
export TRAIN_IMAGES=30000
export BATCH_SIZE=1
export NUM_EPOCHS=10
export STEPS_PER_EPOCH=$((TRAIN_IMAGES / BATCH_SIZE))
export RUN_NAME=soa-production

mkdir -p /srv/facedancer/logs/${RUN_NAME}

tmux new-session -d -s facedancer-production \
  "devenv shell -- python train/train.py \
    --data_dir '/srv/facedancer/tfrecords/target/train/target_train_*-of-*.records' \
    --source_data_dir '/srv/facedancer/tfrecords/source-occluded/train/source_occluded_train_*-of-*.records' \
    --eval_dir '/srv/facedancer/tfrecords/target/validation/target_validation_*-of-*.records' \
    --eval_source_dir '/srv/facedancer/tfrecords/source-occluded/validation/source_occluded_validation_*-of-*.records' \
    --arcface_path /srv/facedancer/models/ArcFace-Res50.h5 \
    --eval_model_expface /srv/facedancer/models/ExpressionEmbedder-B0.h5 \
    --hand_task_path /srv/facedancer/models/hand_landmarker.task \
    --batch_size ${BATCH_SIZE} --eval_batch_size 1 \
    --iterations_per_epoch ${STEPS_PER_EPOCH} --num_epochs ${NUM_EPOCHS} \
    --checkpoint_interval 10000 --device_id 0 \
    --log_dir /srv/facedancer/logs \
    --chkp_dir /srv/facedancer/checkpoints \
    --log_name ${RUN_NAME} \
    2>&1 | tee /srv/facedancer/logs/${RUN_NAME}/training.log"
```

Attach to the native session with `tmux attach -t facedancer-production`.
Monitor TensorBoard separately:

```shell
devenv shell -- tensorboard \
  --logdir /srv/facedancer/logs/soa-production \
  --host 0.0.0.0 --port 6006
```

To resume production, set `LOAD` to a completed checkpoint ID and set
`NUM_EPOCHS` to the new total epoch target, not the number of additional
epochs. Keep `STEPS_PER_EPOCH` exactly equal to the saved
`iterations_per_epoch`:

```shell
export LOAD=30000
export TRAIN_IMAGES=30000
export BATCH_SIZE=1
export NUM_EPOCHS=10
export STEPS_PER_EPOCH=$((TRAIN_IMAGES / BATCH_SIZE))
export RUN_NAME=soa-production

tmux new-session -d -s facedancer-production-resume \
  "devenv shell -- python train/train.py \
    --data_dir '/srv/facedancer/tfrecords/target/train/target_train_*-of-*.records' \
    --source_data_dir '/srv/facedancer/tfrecords/source-occluded/train/source_occluded_train_*-of-*.records' \
    --eval_dir '/srv/facedancer/tfrecords/target/validation/target_validation_*-of-*.records' \
    --eval_source_dir '/srv/facedancer/tfrecords/source-occluded/validation/source_occluded_validation_*-of-*.records' \
    --arcface_path /srv/facedancer/models/ArcFace-Res50.h5 \
    --eval_model_expface /srv/facedancer/models/ExpressionEmbedder-B0.h5 \
    --hand_task_path /srv/facedancer/models/hand_landmarker.task \
    --batch_size ${BATCH_SIZE} --eval_batch_size 1 \
    --iterations_per_epoch ${STEPS_PER_EPOCH} --num_epochs ${NUM_EPOCHS} \
    --checkpoint_interval 10000 --device_id 0 \
    --log_dir /srv/facedancer/logs \
    --chkp_dir /srv/facedancer/checkpoints \
    --log_name ${RUN_NAME} --load ${LOAD} \
    2>&1 | tee -a /srv/facedancer/logs/${RUN_NAME}/training.log"
```

Checkpoint IDs are exact completed iteration counts. Each coherent checkpoint
contains shared `gen/gen.json` and `dis/dis.json` architecture files, ID-scoped
`gen/gen_<ID>.h5` and `dis/dis_<ID>.h5` weights, and
`state/<ID>.json` loop metadata under the selected `log_name`. The state file is
written last and marks a complete generator/discriminator pair. Legacy state
files without the versioned completed-iteration schema are intentionally not
resumable because their periodic and final iteration meanings conflict. A
completed checkpoint ID is immutable; resume from the latest checkpoint or use
a distinct `log_name` rather than overwriting an existing ID. The shared model
architecture files are also immutable after their first save, so every H5 ID
under a run remains loadable with the adjacent JSON architecture.

Resume restores H5 weights and the next loop position. Adam optimizers are
reconstructed without their moment estimates or optimizer iteration counters;
the learning-rate position is reconstructed from the completed global
iteration. Dataset iterators, shuffle order, and random-number state are also
recreated, so resume is step-coherent but not bit-for-bit deterministic.

To export a loaded generator, rerun `train.py` with `--load <ID> --export True`.
Export writes a TensorFlow SavedModel directory at
`exports/<log_name>/facedancer_<ID>`, not a single H5 file.


## PyTorch Implementation
Currently I am working on a PyTorch version of FaceDancer. The training and network code is kind of done. Currently the behaviour compare to TensorFlow is drastically different. Some interesting notes is that the mapping network does not allow for the FaceDancer to learn its task. In current state it provides decent results with the mapping network ommited. I will post the PyTorch version as soon as these issues is diagnosed and resolved.


## Docker 
  * build:
  ```docker build --rm -t faceswap .```
  
  * run:
  ```docker run --gpus all --rm -it -p 8973:8000 -v $(pwd)/results:/workspace/results faceswap```
  
  
## License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png"/></a>

**FaceDancer** is licensed under [Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/)


## Citation
If you use this repository in your work, please cite us:
```
@InProceedings{Rosberg_2023_WACV,
    author    = {Rosberg, Felix and Aksoy, Eren Erdal and Alonso-Fernandez, Fernando and Englund, Cristofer},
    title     = {FaceDancer: Pose- and Occlusion-Aware High Fidelity Face Swapping},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {3454-3463}
}
```

### TODO:
- [ ] Add complete code for calculating IFSR.
- [ ] Add code for all evaluation steps.
- [x] Provide download links to pretrained models.
- [x] Image swap script.
- [ ] Debugging?
