## Model installation, training and evaluation

### Installation
- Python version == 3.8.13

```
    conda create -n nlu python=3.8 -y
    conda activate nlu
    pip install -e . --upgrade --use-feature=in-tree-build
```

### Folder Tree
BKAI Folder Tree (Data Folder):
- BKAI
    + word-level
      ~ train
        + seq.in
        + label
        + seq.out
      ~ dev
        + seq.in
        + label
        + seq.out
      ~ test
        + seq.in
        + label (might not)
        + seq.out (might not)
      ~ intent_label.txt
      ~ slot_label.txt

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

### Training and Evaluation
usage: train.py [-h]
                [--model_dir MODEL_DIR]
                [--data_dir DATA_DIR]
                [--intent_label_file INTENT_LABEL_FILE]
                [--slot_label_file SLOT_LABEL_FILE]
                [--rule_file RULE_FILE]
                [--model_type MODEL_TYPE]
                [--tuning_metric TUNING_METRIC]
                [--seed SEED]
                [--train_batch_size TRAIN_BATCH_SIZE]
                [--max_seq_len MAX_SEQ_LEN]
                [--learning_rate LEARNING_RATE]
                [--num_train_epochs NUM_TRAIN_EPOCHS]
                [--weight_decay WEIGHT_DECAY]
                [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                [--adam_epsilon ADAM_EPSILON]
                [--max_grad_norm MAX_GRAD_NORM]
                [--max_steps MAX_STEPS]
                [--warmup_steps WARMUP_STEPS]
                [--dropout_rate DROPOUT_RATE]
                [--logging_steps LOGGING_STEPS]
                [--save_steps SAVE_STEPS]
                [--do_train]
                [--do_eval]
                [--do_eval_dev]
                [--no_cuda]
                [--ignore_index IGNORE_INDEX]
                [--intent_loss_coef INTENT_LOSS_COEF]
                [--token_level TOKEN_LEVEL]
                [--early_stopping EARLY_STOPPING]
                [--gpu_id GPU_ID]
                [--use_crf]
                [--pretrained]
                [--pretrained_path PRETRAINED_PATH]
                [--use_rule_based]
                [--use_filter]
                [--attention_embedding_size ATTENTION_EMBEDDING_SIZE]
                [--slot_pad_label SLOT_PAD_LABEL]
                [--embedding_type EMBEDDING_TYPE]
                [--use_attention_mask]
                [--train_type TRAIN_TYPE]
                [--val_type VAL_TYPE]
                [--test_type TEST_TYPE]


### Inference
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
        python Smart-Home-Intent-Detection-and-Slot-Filling/augment_data.py --dataset-path ./BKAI/word-level --trainset train/seq.in \
        --train_intent_label train/label --train_slot_label train/seq.out --valset dev/seq.in \
        --val_intent_label dev/label --val_slot_label dev/seq.out \
        --intent-label-file intent_label.txt --slot-label-file slot_label.txt \
        -k 7 --seed 11
    ```

2. Generate filtering model
```
    rm ./BKAI/cached_*

    python3 Smart-Home-Intent-Detection-and-Slot-Filling/train.py --token_level word-level --model_type phobert \
    --model_dir ./drive/MyDrive/Colab/trained_models/filtering_model_1 --data_dir BKAI \
    --intent_label_file intent_label.txt --slot_label_file slot_label.txt \
    --rule_file ./BKAI/rule.csv --train_type train_val \
    --val_type augment_val --test_type augment_val --seed 7 \
    --do_train --num_train_epochs 50 --tuning_metric mean_intent_slot \
    --use_crf --gpu_id 0 --embedding_type soft --intent_loss_coef 0.7 \
    --learning_rate 0.00005 --dropout_rate 0.5 --attention_embedding_size 256 \
    --save_steps 69 --logging_steps 69 --train_batch_size 32
```
    
   3. Filtering
    ```
        python Smart-Home-Intent-Detection-and-Slot-Filling/filter_data.py --input_file ./BKAI/word-level/dev/seq.in \
        --model_dir ./drive/MyDrive/Colab/trained_models/filtering_model_1 --output_dir ./drive/MyDrive/Colab --batch_size 256 \
        --intent_entropy_threshold 0.004 --slot_entropy_threshold 1.0 --max_collect_num 10000
        python Smart-Home-Intent-Detection-and-Slot-Filling/filter_data.py --input_file ./BKAI/word-level/train/seq.in \
        --model_dir ./drive/MyDrive/Colab/trained_models/filtering_model_1 --output_dir ./drive/MyDrive/Colab --batch_size 256 \
        --intent_entropy_threshold 0.004 --slot_entropy_threshold 100 --max_collect_num 10000
    ```

    The label statistic is saved in ./output/filtered_reports.json

   4.  Generate New Augment Data
    ```
        python Smart-Home-Intent-Detection-and-Slot-Filling/augment_data.py --dataset-path ./BKAI/word-level --trainset train/filtered_seq.in \
        --train_intent_label train/filtered_label --train_slot_label train/filtered_seq.out --valset dev/filtered_seq.in \
        --val_intent_label dev/filtered_label --val_slot_label dev/filtered_seq.out \
        --intent-label-file intent_label.txt --slot-label-file slot_label.txt \
        -k 7 --seed 23
    ```
   5.  Make rule matrix
    ```
       python3 Smart-Home-Intent-Detection-and-Slot-Filling/make_rules.py --model_dir ./drive/MyDrive/Colab/trained_models/filtering_model
    ```

   6.  Training model
    ```
        rm ./BKAI/cached_*

        !python3 Smart-Home-Intent-Detection-and-Slot-Filling/train.py --token_level word-level --model_type phobert \
        --model_dir ./drive/MyDrive/Colab/trained_models --data_dir BKAI \
        --intent_label_file intent_label.txt --slot_label_file slot_label.txt \
        --rule_file ./BKAI/rule.csv --train_type augment_train_val_plus \
        --val_type augment_val --test_type test --seed 11 --early_stopping 50\
        --do_train --do_eval --num_train_epochs 500 --tuning_metric intent_acc \
        --use_crf --gpu_id 0 --embedding_type soft --intent_loss_coef 0.7 \
        --learning_rate 0.00005 --dropout_rate 0.7 --attention_embedding_size 384 \
        --save_steps 50 --logging_steps 50 --use_rule_based --train_batch_size 32 
    ```
    7.  Inference
    ```
        python Smart-Home-Intent-Detection-and-Slot-Filling/inference.py --input_file ./BKAI/word-level/test/seq.in \
                --output_file results.csv \
                --model_dir ./drive/MyDrive/Colab/trained_models \
                --batch_size 256
    ```
