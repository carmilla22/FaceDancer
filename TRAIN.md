1. Split source dataset
```bash
python dataset/split_dataset.py --input_dir "assets/source" --output_dir "assets/source_splitted"
```
```bash
python generate_hand_occluded_faces.py --faces-dir assets/dataset/celeb/CelebA-HQ-img --masks-dir assets/dataset/hands_masks --hands-dir assets/dataset/hands --output assets/dataset/celeb_occluded --num-faces 0 --variants-per-face 4 --overwrite
```
2. Split target dataset
```bash
python dataset/split_dataset.py --input_dir "assets/target" --output_dir "assets/target_splitted"
```
3. Process source dataset
```bash
python dataset/dataset_sharding.py --data_dir "assets/source_splitted" --target_dir "assets/source_processed/validation" --data_name "source_occluded" --data_type "validation" --num_shards 20 --max_images 20 && python dataset/dataset_sharding.py --data_dir "assets/source_splitted" --target_dir "assets/source_processed/train" --data_name "source_occluded" --data_type "train" --num_shards 20 --max_images 20
```
4. Process target dataset
```bash
python dataset/dataset_sharding.py --data_dir "assets/target_splitted" --target_dir "assets/target_processed/validation" --data_name "target_occluded" --data_type "validation" --num_shards 20 --max_images 20 && python dataset/dataset_sharding.py --data_dir "assets/target_splitted" --target_dir "assets/target_processed/train" --data_name "target_occluded" --data_type "train" --num_shards 20 --max_images 20
```
5. Start the train
```bash
python train/train.py --data_dir "assets/target_processed/train/target_occluded_train_*-of-*.records" --eval_dir "assets/target_processed/validation/target_occluded_validation_*-of-*.records" --source_data_dir "assets/source_processed/train/source_occluded_train_*-of-*.records" --eval_source_dir "assets/source_processed/validation/source_occluded_validation_*-of-*.records" --hand_task_path "models/hand_landmarker.task" --arcface_path "arcface_model/ArcFace-Res50.h5" --eval_model_expface "expressionembedder_model/ExpressionEmbedder-B0.h5" --batch_size 8 --num_epochs 100 --iterations_per_epoch 10000 --log_dir "logs/runs" --chkp_dir "checkpoints" --log_name "facedancer_gpu_100_epochs_10000_iterations"
```
6. [OPTIONAL] Resume training
```bash
python train/train.py --data_dir "assets/target_processed/train/target_occluded_train_*-of-*.records" --eval_dir "assets/target_processed/validation/target_occluded_validation_*-of-*.records" --source_data_dir "assets/source_processed/train/source_occluded_train_*-of-*.records" --eval_source_dir "assets/source_processed/validation/source_occluded_validation_*-of-*.records" --hand_task_path "models/hand_landmarker.task" --arcface_path "arcface_model/ArcFace-Res50.h5" --eval_model_expface "expressionembedder_model/ExpressionEmbedder-B0.h5" --batch_size 8 --num_epochs 100 --iterations_per_epoch 10000 --log_dir "logs/runs" --chkp_dir "checkpoints" --log_name "facedancer_gpu_100_epochs_10000_iterations" --load CHECKPOINT_ID
```
