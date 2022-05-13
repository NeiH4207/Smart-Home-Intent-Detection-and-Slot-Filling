import argparse
import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from src.utils import MODEL_CLASSES, get_intent_labels, get_slot_labels, init_logger, load_tokenizer, set_seed
from scipy.stats import entropy

import matplotlib.pyplot as plt

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


def filter(pred_config):
    # load model and args
    args = get_args(pred_config)
    set_seed(args)
    device = get_device(pred_config)
    model = load_model(pred_config, args, device)
    logger.info(args)
    label_file = '/'.join(pred_config.input_file.split('/')[:-1]) + '/label'
    slots_file = '/'.join(pred_config.input_file.split('/')[:-1]) + '/seq.out'
    labels = []
    slots = []
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            labels.append(line)
    with open(slots_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            slots.append(line)
            
    intent_label_lst = get_intent_labels(args)
    slot_label_lst = get_slot_labels(args)
            
    # Convert input file to TensorDataset
    pad_token_label_id = args.ignore_index
    tokenizer = load_tokenizer(args)
    lines = read_input_file(pred_config)
    dataset = convert_input_file_to_tensor_dataset(lines, pred_config, args, tokenizer, pad_token_label_id)

    # filter
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=pred_config.batch_size)

    intent_entropies = []
    slot_entropies = []

    for batch in tqdm(data_loader, desc="filtering"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "intent_label_ids": None,
                "slot_labels_ids": None,
            }
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = batch[2]
            outputs = model(**inputs)
            _, (intent_logits, slot_logits) = outputs[:2]

            intent_probs = np.exp(intent_logits.detach().cpu().numpy())
            entropies = entropy(intent_probs, axis=1)
            intent_entropies.extend(entropies)
            slot_probs = np.exp(slot_logits.detach().cpu().numpy())
            entropies = np.mean(entropy(slot_probs, axis=2), axis=1)
            slot_entropies.extend(entropies)
    
    # plot entropies two columns graph
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(intent_entropies, label="intent")
    ax.legend()
    plt.savefig(pred_config.output_dir + "/intent_entropy.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(slot_entropies, label="slot")
    ax.legend()
    plt.savefig(pred_config.output_dir + "/slot_entropy.png")
    plt.close(fig)
    # Write to output file
    intent_entropy_threshold = pred_config.intent_entropy_threshold
    slot_entropy_threshold = pred_config.slot_entropy_threshold

    filter_path = '/'.join(pred_config.input_file.split('/')[:-1]) 
    filter_intent_input_file = filter_path + '/filtered_seq.in'
    filter_intent_label_lst_file = filter_path + '/filtered_label'
    filter_slot_input_file = filter_path + '/filtered_seq.out'
    
    before_intent_filtered_reports = {}
    before_slot_filtered_reports = {}
    
    intent_filtered_reports = {}
    slot_filtered_reports = {}
    
    for intent_label in intent_label_lst:
        intent_filtered_reports[intent_label] = 0
        before_intent_filtered_reports[intent_label] = 0
    
    for slot_label in slot_label_lst:
        slot_filtered_reports[slot_label] = 0
        before_slot_filtered_reports[slot_label] = 0
        
    max_collect_num = 1800
        
    with open(filter_intent_input_file, 'w') as f_intent_input, \
            open(filter_intent_label_lst_file, 'w') as f_intent_label_lst, \
            open(filter_slot_input_file, 'w') as f_slot_input:
        for i in range(len(intent_entropies)):
            temp_intent_entropy_threshold = intent_entropy_threshold
            temp_slot_entropy_threshold = slot_entropy_threshold
            temp_max_collect_num = max_collect_num
            if labels[i] in ['smart.home.decrease.level']:
                temp_intent_entropy_threshold *= 2
                temp_slot_entropy_threshold *= 2
                temp_max_collect_num *= 1.2
            if labels[i] in ['smart.home.increase.level', 'smart.home.set.level']:
                temp_intent_entropy_threshold *= 1.5
                temp_slot_entropy_threshold *= 1.5
                temp_max_collect_num *= 1.1
            if intent_filtered_reports[labels[i]] > temp_max_collect_num:
                continue
            if labels[i] == 'greeting' or intent_entropies[i] < temp_intent_entropy_threshold and \
                    (slot_entropies[i] < temp_slot_entropy_threshold or\
                        'statusstatus' in slots[i] or\
                        'I-allall' in slots[i] 
                        ):
                f_intent_input.write(' '.join(lines[i]) + '\n')
                f_intent_label_lst.write(labels[i] + '\n')
                f_slot_input.write(slots[i] + '\n')
                intent_filtered_reports[labels[i]] += 1    
                for slot in slots[i].split(' '):
                    slot_filtered_reports[slot] += 1
                    
            before_intent_filtered_reports[labels[i]] += 1 
            for slot in slots[i].split(' '):
                before_slot_filtered_reports[slot] += 1
    
    # save reports
    with open(pred_config.output_dir + "/reports.json", 'w') as f:
        json.dump(before_intent_filtered_reports, f, indent=4)
    with open(pred_config.output_dir + "/filtered_reports.json", 'w') as f:
        json.dump(intent_filtered_reports, f, indent=4)
        
    with open(pred_config.output_dir + "/slot_reports.json", 'w') as f:
        json.dump(before_slot_filtered_reports, f, indent=4)
    with open(pred_config.output_dir + "/filtered_slot_reports.json", 'w') as f:
        json.dump(slot_filtered_reports, f, indent=4)
    logger.info("filterion Done!")


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default="BKAI/word-level/augment_train_val_plus/seq.in", type=str, help="Input file for filterion")
    parser.add_argument("--output_file", default="output/results.csv", type=str, help="Output file for filterion")
    parser.add_argument("--model_dir", default="./models/filtering_model", type=str, help="Path to save, load model")

    parser.add_argument("--batch_size", default=128, type=int, help="Batch size for filterion")
    parser.add_argument("--intent_entropy_threshold", default=0.25, type=float, help="Entropy intent threshold")
    parser.add_argument("--slot_entropy_threshold", default=0.45, type=float, help="Entropy slot threshold")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    parser.add_argument("--output_dir", default="output/", type=str, help="Output file for filterion")
    pred_config = parser.parse_args()
    filter(pred_config)
