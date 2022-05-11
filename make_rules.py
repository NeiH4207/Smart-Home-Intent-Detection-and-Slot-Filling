import argparse
import csv
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from utils import MODEL_CLASSES, get_intent_labels, get_slot_labels, init_logger, load_tokenizer


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
            args.model_dir, args=args, intent_label_lst=get_intent_labels(args), slot_label_lst=get_slot_labels(args)
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


def predict(pred_config):
    # load model and args
    args = get_args(pred_config)
    logger.info(args)

    slot_label_lst = get_slot_labels(args)
    n_slot_labels = len(slot_label_lst)
    rule_matrix = np.ones((n_slot_labels, n_slot_labels), dtype=np.int)
    count = 0
    for i in range(n_slot_labels):
        for j in range(n_slot_labels):
            if slot_label_lst[i][0] == 'B' and slot_label_lst[j][0] == 'I' and \
                slot_label_lst[i][2:] != slot_label_lst[j][2:]:
                rule_matrix[i, j] = 0
                print(slot_label_lst[i], slot_label_lst[j])
            elif slot_label_lst[i][0] == 'B' and slot_label_lst[j][0] == 'B' and \
                slot_label_lst[i][2:] == slot_label_lst[j][2:]:
                rule_matrix[i, j] = 0
                print(slot_label_lst[i], slot_label_lst[j])
            elif slot_label_lst[i][0] == 'O' and slot_label_lst[j][0] == 'I':
                rule_matrix[i, j] = 0
                print(slot_label_lst[i], slot_label_lst[j])
            elif slot_label_lst[i][0] == 'I' and slot_label_lst[j][0] == 'I' and \
                slot_label_lst[i][2:] != slot_label_lst[j][2:]:
                rule_matrix[i, j] = 0
                print(slot_label_lst[i], slot_label_lst[j])
    print(np.sum(rule_matrix))
    # save to csv 
    with open('BKAI/rule.csv', 'w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(rule_matrix)
    


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./trained_models", type=str, help="Path to save, load model")
    pred_config = parser.parse_args()
    predict(pred_config)
