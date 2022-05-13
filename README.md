## Model installation, training and evaluation

### Installation
- Python version == 3.8.13

```
    conda create -n nlu python=3.8 -y
    conda activate nlu
    pip install -e . --upgrade --use-feature=in-tree-build
```

### Augmentation
usage: augment_data.py [-h] [--dataset-path DATASET_PATH]
                       [--trainset TRAINSET]
                       [--train_intent_label TRAIN_INTENT_LABEL]
                       [--train_slot_label TRAIN_SLOT_LABEL]
                       [--valset VALSET]
                       [--val_intent_label VAL_INTENT_LABEL]
                       [--val_slot_label VAL_SLOT_LABEL]
                       [--intent-label-file INTENT_LABEL_FILE]
                       [--slot-label-file SLOT_LABEL_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --dataset-path DATASET_PATH
                        path to the dataset
  --trainset TRAINSET   name of the training dataset file
  --train_intent_label TRAIN_INTENT_LABEL
                        name of the training intent label file
  --train_slot_label TRAIN_SLOT_LABEL
                        name of the training slot label file
  --valset VALSET       name of the validation dataset file
  --val_intent_label VAL_INTENT_LABEL
                        name of the validation intent label file
  --val_slot_label VAL_SLOT_LABEL
                        name of the validation slot label file
  --intent-label-file INTENT_LABEL_FILE
                        name of the intent label file
  --slot-label-file SLOT_LABEL_FILE
                        name of the slot label file

Example:               
```
    python augment_data.py --dataset-path ./BKAI/word-level --trainset train/seq.in --train_intent_label train/label --train_slot_label train/seq.out --valset val/seq.in --val_intent_label val/label --val_slot_label val/seq.out --intent-label-file train/intent_labels.txt --slot-label-file train/slot_labels.txt
```
### Filtering
usage: filter_data.py [-h] [--input_file INPUT_FILE] [--output_file OUTPUT_FILE]
                      [--model_dir MODEL_DIR] [--batch_size BATCH_SIZE]
                      [--intent_entropy_threshold INTENT_ENTROPY_THRESHOLD]
                      [--slot_entropy_threshold SLOT_ENTROPY_THRESHOLD] [--no_cuda]
                      [--output_dir OUTPUT_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE
                        Input file for filterion
  --model_dir MODEL_DIR
                        Path to save, load model
  --batch_size BATCH_SIZE
                        Batch size for filterion
  --intent_entropy_threshold INTENT_ENTROPY_THRESHOLD
                        Entropy intent threshold
  --slot_entropy_threshold SLOT_ENTROPY_THRESHOLD
                        Entropy slot threshold
  --no_cuda             Avoid using CUDA when available
  --output_dir OUTPUT_DIR
                        Output file for filterion

Example:
```
    python filter_data.py --input_file ./BKAI/word-level/augment_train_val_plus/seq.in --model_dir ./models/filtering_model --batch_size 32 --intent_entropy_threshold 0.75 --slot_entropy_threshold 0.75
```

### Training and Evaluation
```
    python3 main.py --token_level word-level --model_type phobert --model_dir ./trained_models --data_dir BKAI --seed 11 --do_train --do_eval --save_steps 69 --logging_steps 69 --num_train_epochs 50 --train_batch_size 128 --tuning_metric mean_intent_slot --use_crf --gpu_id 0 --embedding_type soft --intent_loss_coef 0.7 --learning_rate 0.00004 --dropout_rate 0.7 --attention_embedding_size 384 --save_steps 98 --logging_steps 98 --use_rule_based --use_filter
```


### Inference
usage: inference [-h] [--input_file INPUT_FILE]
                 [--output_file OUTPUT_FILE]
                 [--model_dir MODEL_DIR]
                 [--batch_size BATCH_SIZE] [--no_cuda]

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE
                        Input file for prediction
  --output_file OUTPUT_FILE
                        Output file for prediction
  --model_dir MODEL_DIR
                        Path to save, load model
  --batch_size BATCH_SIZE
                        Batch size for prediction
  --no_cuda             Avoid using CUDA when available

Example:
```
    inference --input_file ./BKAI/word-level/test/seq.in \
                --output_file ./output/results.csv \
                --model_dir ./trained_models \
                --batch_size 32
```

# Pipelines
   1. Augmentation
    ```
        python augment_data.py --dataset-path ./BKAI/word-level --trainset train/seq.in \
        --train_intent_label train/label --train_slot_label train/seq.out --valset dev/seq.in \
        --val_intent_label dev/label --val_slot_label dev/seq.out \
        --intent-label-file intent_label.txt --slot-label-file slot_label.txt
    ```

   2. Generate filtering model
    ```
        python3 train.py --token_level word-level --model_type phobert \
        --model_dir ./trained_models/filtering_model --data_dir BKAI \
        --intent_label_file intent_label.txt --slot_label_file slot_label.txt \
        --rule_file ./BKAI/rule.csv --train_type augment_train_val_plus \
        --val_type augment_val test_type augment_val --seed 11 \
        --do_train --do_eval --num_train_epochs 50 --tuning_metric mean_intent_slot \
        --use_crf --gpu_id 0 --embedding_type soft --intent_loss_coef 0.7 \
        --learning_rate 0.00004 --dropout_rate 0.7 --attention_embedding_size 384 \
        --save_steps 69 --logging_steps 69 --use_rule_based --train_batch_size 32
    ```
   4. Filtering
    ```
        python filter_data.py --input_file ./BKAI/word-level/augment_train_val_plus/seq.in \
        --model_dir ./trained_models/filtering_model --batch_size 32 \
        --intent_entropy_threshold 0.75 --slot_entropy_threshold 0.75
        python filter_data.py --input_file ./BKAI/word-level/augment_val/seq.in \
        --model_dir ./trained_models/filtering_model --batch_size 32 \
        --intent_entropy_threshold 0.75 \
        --slot_entropy_threshold 0.75
    ```

    The label statistic is saved in ./output/filtered_reports.json

   5.  Make rule matrix
    ```
        python3 make_rules.py  --model_dir ./trained_models/filtering_model
    ```

   6.  Training model
    ```
        python3 train.py --token_level word-level --model_type phobert \
        --model_dir ./trained_models --data_dir BKAI \
        --intent_label_file intent_label.txt --slot_label_file slot_label.txt \
        --rule_file ./BKAI/rule.csv --train_type augment_train_val_plus \
        --val_type augment_val --test_type test \
        --seed 11 --do_train --do_eval --num_train_epochs 50 \
        --tuning_metric mean_intent_slot --use_crf --gpu_id 0 \
        --embedding_type soft --intent_loss_coef 0.7 \
        --learning_rate 0.00004 --dropout_rate 0.7 \
        --attention_embedding_size 384 --save_steps 69 \
        --logging_steps 69 --use_rule_based --train_batch_size 64 \
        --use_filter
    ```