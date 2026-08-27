1. Split source dataset
```bash
python dataset/split_dataset.py --input_dir "assets/source" --output_dir "assets/source_splitted"
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
