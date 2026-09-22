# FaceDancer Training

This guide sets up and runs the project on an x86_64 Linux machine or WSL2.

## 1. Install Nix

Install Nix in multi-user mode:

```bash
sh <(curl -L https://nixos.org/nix/install) --daemon
```

On WSL without systemd, replace `--daemon` with `--no-daemon`.

Restart the terminal, then enable the Nix features used by devenv:

```bash
mkdir -p ~/.config/nix
printf 'experimental-features = nix-command flakes\n' >> ~/.config/nix/nix.conf
nix --version
```

## 2. Install direnv and devenv

```bash
nix profile install nixpkgs#direnv
nix profile install --accept-flake-config github:cachix/devenv/latest
```

## 3. Enable Automatic Environment Loading

For Bash, which is the default shell on most Linux and WSL installations:

```bash
printf '\neval "$(direnv hook bash)"\n' >> ~/.bashrc
exec bash
```

For Zsh, use `~/.zshrc` and `direnv hook zsh` instead.

Enter the project and approve its environment:

```bash
cd FaceDancer
direnv allow
```

The first load downloads and builds the pinned dependencies, models, and data,
so it can take a while. Later visits load the environment automatically.

## 4. Check the Environment

For GPU training, install the NVIDIA driver on the host first. On WSL2, install
the NVIDIA WSL driver on Windows. Then verify the environment:

```bash
nvidia-smi
fd-check-env
```

`fd-check-env` must report the NVIDIA GPU and a successful TensorFlow GPU matrix
multiplication.

## 5. Prepare Data and Train

Run these commands from the project root with the devenv environment loaded.

### 5.1. Split the Source Dataset

```bash
python dataset/split_dataset.py \
  --input_dir "assets/source" \
  --output_dir "assets/source_splitted"
```

Generate the hand-occluded source images:

```bash
python generate_hand_occluded_faces.py \
  --faces-dir assets/dataset/celeb/CelebA-HQ-img \
  --masks-dir assets/dataset/hands_masks \
  --hands-dir assets/dataset/hands \
  --output assets/dataset/celeb_occluded \
  --num-faces 0 \
  --variants-per-face 4 \
  --overwrite
```

### 5.2. Split the Target Dataset

```bash
python dataset/split_dataset.py \
  --input_dir "assets/target" \
  --output_dir "assets/target_splitted"
```

### 5.3. Process the Source Dataset

```bash
python dataset/dataset_sharding.py \
  --data_dir "assets/source_splitted" \
  --target_dir "assets/source_processed/validation" \
  --data_name "source_occluded" \
  --data_type "validation" \
  --num_shards 20 \
  --max_images 20 &&
python dataset/dataset_sharding.py \
  --data_dir "assets/source_splitted" \
  --target_dir "assets/source_processed/train" \
  --data_name "source_occluded" \
  --data_type "train" \
  --num_shards 20 \
  --max_images 20
```

### 5.4. Process the Target Dataset

```bash
python dataset/dataset_sharding.py \
  --data_dir "assets/target_splitted" \
  --target_dir "assets/target_processed/validation" \
  --data_name "target_occluded" \
  --data_type "validation" \
  --num_shards 20 \
  --max_images 20 &&
python dataset/dataset_sharding.py \
  --data_dir "assets/target_splitted" \
  --target_dir "assets/target_processed/train" \
  --data_name "target_occluded" \
  --data_type "train" \
  --num_shards 20 \
  --max_images 20
```

### 5.5. Start Training

```bash
python train/train.py \
  --data_dir "assets/target_processed/train/target_occluded_train_*-of-*.records" \
  --eval_dir "assets/target_processed/validation/target_occluded_validation_*-of-*.records" \
  --source_data_dir "assets/source_processed/train/source_occluded_train_*-of-*.records" \
  --eval_source_dir "assets/source_processed/validation/source_occluded_validation_*-of-*.records" \
  --hand_task_path "models/hand_landmarker.task" \
  --arcface_path "arcface_model/ArcFace-Res50.h5" \
  --eval_model_expface "expressionembedder_model/ExpressionEmbedder-B0.h5" \
  --batch_size 8 \
  --num_epochs 100 \
  --iterations_per_epoch 10000 \
  --log_dir "logs/runs" \
  --chkp_dir "checkpoints" \
  --log_name "facedancer_gpu_100_epochs_10000_iterations"
```

### 5.6. Resume Training

Replace `CHECKPOINT_ID` with the checkpoint to resume:

```bash
python train/train.py \
  --data_dir "assets/target_processed/train/target_occluded_train_*-of-*.records" \
  --eval_dir "assets/target_processed/validation/target_occluded_validation_*-of-*.records" \
  --source_data_dir "assets/source_processed/train/source_occluded_train_*-of-*.records" \
  --eval_source_dir "assets/source_processed/validation/source_occluded_validation_*-of-*.records" \
  --hand_task_path "models/hand_landmarker.task" \
  --arcface_path "arcface_model/ArcFace-Res50.h5" \
  --eval_model_expface "expressionembedder_model/ExpressionEmbedder-B0.h5" \
  --batch_size 8 \
  --num_epochs 100 \
  --iterations_per_epoch 10000 \
  --log_dir "logs/runs" \
  --chkp_dir "checkpoints" \
  --log_name "facedancer_gpu_100_epochs_10000_iterations" \
  --load CHECKPOINT_ID
```

If automatic loading is unavailable, run any project command explicitly with:

```bash
devenv shell -- <command>
```

For best WSL performance, keep the repository and datasets in the Linux
filesystem, such as `~/FaceDancer`, rather than under `/mnt/c`.
