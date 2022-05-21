import argparse
import csv
import logging
import os

import numpy as np
import torch
from torch.utils.data import TensorDataset
from src.utils import MODEL_CLASSES, get_intent_labels, get_slot_labels, init_logger, load_tokenizer


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

def make_rule(pred_config):
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
                # print(slot_label_lst[i], slot_label_lst[j])
            elif slot_label_lst[i][0] == 'B' and slot_label_lst[j][0] == 'B' and \
                slot_label_lst[i][2:] == slot_label_lst[j][2:]:
                rule_matrix[i, j] = 0
                # print(slot_label_lst[i], slot_label_lst[j])
            elif slot_label_lst[i][0] == 'O' and slot_label_lst[j][0] == 'I':
                rule_matrix[i, j] = 0
                # print(slot_label_lst[i], slot_label_lst[j])
            elif slot_label_lst[i][0] == 'I' and slot_label_lst[j][0] == 'I' and \
                slot_label_lst[i][2:] != slot_label_lst[j][2:]:
                rule_matrix[i, j] = 0
                # print(slot_label_lst[i], slot_label_lst[j])
            # I-* == B-*
            elif slot_label_lst[i][0] == 'I' and slot_label_lst[j][0] == 'B' and \
                slot_label_lst[i][2:] == slot_label_lst[j][2:]:
                rule_matrix[i, j] = 0
            elif slot_label_lst[i] == "B-devicedevice" and slot_label_lst[j] == "B-sysnumbersysnumber":
                rule_matrix[i, j] = 0
                # print(slot_label_lst[i], slot_label_lst[j])
                
    print(np.sum(rule_matrix))
    # save to csv 
    with open(pred_config.outfile, 'w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(rule_matrix)
    


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./trained_models/filtering_model", type=str, help="Path to save, load model")
    parser.add_argument("--outfile", default="./BKAI/rule.csv", type=str, help="Path to save, load model")
    pred_config = parser.parse_args()
    make_rule(pred_config)
