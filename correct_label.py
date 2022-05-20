import argparse
import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from transformers import RobertaPreTrainedModel
from src.utils import MODEL_CLASSES, get_intent_labels, get_slot_labels, init_logger, load_tokenizer, set_seed
from scipy.stats import entropy

import matplotlib.pyplot as plt
# import KNN
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)


def get_device(pred_config):
    return "cuda" if torch.cuda.is_available() and not pred_config.no_cuda else "cpu"


def get_args(pred_config):
    return torch.load(os.path.join(pred_config.model_dir, "training_args.bin"))

def load_model(pred_config, args, device):
    # Check whether model exists
    if not os.path.exists(pred_config.model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        model = MODEL_CLASSES[args.model_type][1].from_pretrained(
            pred_config.model_dir, args=args, intent_label_lst=get_intent_labels(args), slot_label_lst=get_slot_labels(args)
        )
        model.to(device)
        model.eval()
        logger.info("***** Model Loaded *****")
    except Exception:
        raise Exception("Some model files might be missing...")

    return model


def read_input_file(pred_config):
    lines = []
    with open(pred_config.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            words = line.split()
            lines.append(words)

    return lines


def convert_input_file_to_tensor_dataset(
    lines,
    pred_config,
    args,
    tokenizer,
    pad_token_label_id,
    cls_token_segment_id=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_slot_label_mask = []

    for words in lines:
        tokens = []
        slot_label_mask = []
        for word in words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_label_mask.extend([pad_token_label_id + 1] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > args.max_seq_len - special_tokens_count:
            tokens = tokens[: (args.max_seq_len - special_tokens_count)]
            slot_label_mask = slot_label_mask[: (args.max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)
        slot_label_mask += [pad_token_label_id]

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids
        slot_label_mask = [pad_token_label_id] + slot_label_mask

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = args.max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_label_mask = slot_label_mask + ([pad_token_label_id] * padding_length)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
        all_slot_label_mask.append(slot_label_mask)

    # Change to Tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_slot_label_mask = torch.tensor(all_slot_label_mask, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask)

    return dataset


def correct_label(pred_config):
    # load model and args
    args = get_args(pred_config)
    set_seed(args)
    device = get_device(pred_config)        
    intent_label_lst = get_intent_labels(args)
    slot_label_lst = get_slot_labels(args)
    config_class, model_class, _ = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path, finetuning_task=args.token_level)
    model = model_class.from_pretrained(
        args.model_name_or_path,
        config=config,
        args=args,
        intent_label_lst=intent_label_lst,
        slot_label_lst=slot_label_lst,
            )
    labels = []
    with open(pred_config.label_standard_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            labels.append(line)
    
    noise_labels = []
    with open(pred_config.label_noise_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            noise_labels.append(line)
    
    standard_slot_labels = []
    with open(pred_config.slot_label_standard_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            standard_slot_labels.append(line)
    
    noise_slot_labels = []
    with open(pred_config.slot_label_noise_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            noise_slot_labels.append(line)
    
    tokenizer = load_tokenizer(args)
    logger.info(args)
    noise_data = []
    stardard_data = []
    
    with open(pred_config.input_noise_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            words = line.split()
            noise_data.append(words)
            
    with open(pred_config.input_standard_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            words = line.split()
            stardard_data.append(words)
        
    # convert to tensor
    pad_token_label_id = args.ignore_index
    noise_dataset = convert_input_file_to_tensor_dataset(noise_data, pred_config, args, tokenizer, pad_token_label_id)
    standard_dataset = convert_input_file_to_tensor_dataset(stardard_data, pred_config, args, tokenizer, pad_token_label_id)
    
    model.to(device)
    model.eval()
    noise_features = []
    standard_features = []
    
    with torch.no_grad():
        noise_dataloader = DataLoader(noise_dataset, batch_size=pred_config.batch_size, shuffle=False)
        for i, (input_ids, attention_mask, token_type_ids, slot_label_mask) in enumerate(noise_dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            slot_label_mask = slot_label_mask.to(device)
            features = model.get_features(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            noise_features.append(features)
        standard_dataloader = DataLoader(standard_dataset, batch_size=pred_config.batch_size, shuffle=False)
        for i, (input_ids, attention_mask, token_type_ids, slot_label_mask) in enumerate(standard_dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            slot_label_mask = slot_label_mask.to(device)
            features = model.get_features(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            standard_features.append(features)
            
    # Implement KNN Model to label the noise data
    noise_features = torch.cat(noise_features, dim=0)
    standard_features = torch.cat(standard_features, dim=0)
    noise_features = noise_features.cpu().numpy()
    standard_features = standard_features.cpu().numpy()
    knn = KNeighborsClassifier(n_neighbors=6)
    knn.fit(standard_features, labels)
    pred_labels = knn.predict(noise_features)
    
    corrected_dict = {}
    num_corrected_same_noise = 0
    for i in range(len(pred_labels)):
        key = noise_labels[i] + ' -> '+ pred_labels[i]
        if noise_labels[i] == pred_labels[i]:
            key += ' (correct)'
            num_corrected_same_noise += 1
        if key not in corrected_dict:
            corrected_dict[key] = 1
        else:
            corrected_dict[key] += 1
    print(json.dumps(corrected_dict, indent=4))
    print('num corrected same noise labels: ', num_corrected_same_noise)
    print('Num different noise labels: ', len(noise_labels) - num_corrected_same_noise)
    logger.info("Finish predicting")
    
    # Merge the corrected labels with the original labels and save
    final_label_file = pred_config.label_noise_file.replace('noise', 'final')
    final_input_file = pred_config.input_noise_file.replace('noise', 'final')
    final_slot_label_file = pred_config.slot_label_noise_file.replace('noise', 'final')
    
    with open(final_label_file, 'w', encoding='utf-8') as f:
        for i in range(len(labels)):
            f.write(labels[i]+'\n')
        for i in range(len(pred_labels)):
            f.write(pred_labels[i]+'\n')
            
    with open(final_input_file, 'w', encoding='utf-8') as f:
        for i in range(len(stardard_data)):
            f.write(' '.join(stardard_data[i])+'\n')
        for i in range(len(noise_data)):
            f.write(' '.join(noise_data[i])+'\n')
            
    with open(final_slot_label_file, 'w', encoding='utf-8') as f:
        for i in range(len(standard_slot_labels)):
            f.write(standard_slot_labels[i]+'\n')
        for i in range(len(noise_slot_labels)):
            f.write(noise_slot_labels[i]+'\n')
            
            
    logger.info("Finish saving")
    
if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_noise_file", default="BKAI/word-level/dev/noise_seq.in", type=str, help="Input file need to be corrected")
    parser.add_argument("--label_noise_file", default="BKAI/word-level/dev/noise_label", type=str, help="Label file for training")
    parser.add_argument("--slot_label_noise_file", default="BKAI/word-level/dev/noise_seq.out", type=str, help="Slot label file for training")
    
    parser.add_argument("--input_standard_file", default="BKAI/word-level/dev/filtered_seq.in", type=str, help="Input file for training")
    parser.add_argument("--label_standard_file", default="BKAI/word-level/dev/filtered_label", type=str, help="Label file for training")
    parser.add_argument("--slot_label_standard_file", default="BKAI/word-level/dev/filtered_seq.out", type=str, help="Slot label file for training")
    
    parser.add_argument("--model_dir", default="./trained_models/filtering_model", type=str, help="Path to save, load model")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for filterion")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    pred_config = parser.parse_args()
    correct_label(pred_config)
