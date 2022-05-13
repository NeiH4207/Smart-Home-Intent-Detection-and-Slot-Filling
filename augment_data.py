"""
Created on Mon Mar 11 16:00:47 2022
@author: hien
"""
from __future__ import division
import argparse
from copy import deepcopy
import logging
import math
import os

from matplotlib import pyplot as plt
from src.data_loader import DataLoader
from utils import set_seed
log = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='BKAI/word-level',
                        help='path to the dataset')
    parser.add_argument('--trainset', type=str, default='train/seq.in',
                        help='name of the training dataset file')
    parser.add_argument('--train_intent_label', type=str, default='train/label',
                        help='name of the training intent label file')
    parser.add_argument('--train_slot_label', type=str, default='train/seq.out',
                        help='name of the training slot label file')
    parser.add_argument('--valset', type=str, default='dev/seq.in',
                        help='name of the validation dataset file')
    parser.add_argument('--val_intent_label', type=str, default='dev/label',
                        help='name of the validation intent label file')
    parser.add_argument('--val_slot_label', type=str, default='dev/seq.out',
                        help='name of the validation slot label file')
    parser.add_argument('--intent-label-file', type=str, default='train/intent_label.txt',
                        help='name of the intent label file')
    parser.add_argument('--slot-label-file', type=str, default='train/slot_label.txt',
                        help='name of the slot label file')
    parser.add_argument('-k', '--K', type=int, default=7,
                        help='number of generated examples')
    parser.add_argument('--seed', type=int, default=1,
                        help='number of generated examples')
    
    args = parser.parse_args()
    args.intent_label_file = os.path.join(args.dataset_path, args.intent_label_file)
    args.slot_label_file = os.path.join(args.dataset_path, args.slot_label_file)
    return args


def main():
    args = parse_args()
    set_seed(args)
    data_loader = DataLoader()
    
    intent_label_lst = []
    slot_label_lst = []
    with open(args.intent_label_file, 'r') as f:
        for line in f:
            intent_label_lst.append(line.strip())
    with open(args.slot_label_file, 'r') as f:
        for line in f:
            slot_label_lst.append(line.strip())
    intent_label_lst = sorted(intent_label_lst)
    slot_label_lst = sorted(slot_label_lst)
    
    ''' Load data '''
    data_loader.make(
        dataset_path=args.dataset_path,
        trainset_path=args.trainset,
        train_intent_label=args.train_intent_label,
        train_slot_label=args.train_slot_label,
        valset_path=args.valset,
        val_intent_label=args.val_intent_label,
        val_slot_label=args.val_slot_label,
        intent_label_file=args.intent_label_file,
        slot_label_file=args.slot_label_file)
    
    data_loader.make_dict(data_loader.dataset['train'])
    data_loader.make_dict(data_loader.dataset['val'])
        
    agumented_train_dataset = data_loader.augment(data_loader.dataset['train'],
                                                  merge=True, drop_rate=0.5,
                                                  change_rate=1.0, K=5)
    agumented_val_dataset = data_loader.augment(data_loader.dataset['val'], 
                                                merge=True, drop_rate=0.5, 
                                                change_rate=1.0, K=5)
    agumented_train_val_plus_dataset = deepcopy(agumented_train_dataset)
    
    for sentence, intent, slots in zip(agumented_val_dataset['data'], 
                                       agumented_val_dataset['intent_label'], 
                                       agumented_val_dataset['slot_label']):
        agumented_train_val_plus_dataset['data'].append(sentence)
        agumented_train_val_plus_dataset['intent_label'].append(intent)
        agumented_train_val_plus_dataset['slot_label'].append(slots)
    
    data_loader.dump(path='BKAI/word-level/augment_train',
                     dataset=agumented_train_dataset)
    data_loader.dump(path='BKAI/word-level/augment_train_val_plus',
                     dataset=agumented_train_val_plus_dataset)
    
    agumented_val_dataset = data_loader.augment(data_loader.dataset['val'], 
                                                merge=False, drop_rate=0.5, change_rate=1.0, K=5)
    agumented_train_val_dataset = deepcopy(agumented_train_dataset)
    
    for sentence, intent, slots in zip(agumented_val_dataset['data'],
                                        agumented_val_dataset['intent_label'],
                                        agumented_val_dataset['slot_label']):
        agumented_train_val_dataset['data'].append(sentence)
        agumented_train_val_dataset['intent_label'].append(intent)
        agumented_train_val_dataset['slot_label'].append(slots)
        
    data_loader.dump(path='BKAI/word-level/augment_val',
                        dataset=agumented_val_dataset)
    data_loader.dump(path='BKAI/word-level/augment_train_val',
                        dataset=agumented_train_val_dataset)
                   
                   
                   
    # statistic the number of intent and slot for each label
    intent_label_dict = {}
    slot_label_dict = {}
    for intent_label in intent_label_lst:
        intent_label_dict[intent_label] = 0
    for slot_label in slot_label_lst:
        if slot_label != 'O' and slot_label != 'PAD':
            slot_label_dict[slot_label] = 0
    
    for intent in agumented_train_dataset['intent_label']:
        intent_label_dict[intent] += 1
                
    for slots in agumented_train_dataset['slot_label']:
        slot_list = slots.split(' ')
        for slot in slot_list:
            if slot != 'O' and slot != 'PAD':
                slot_label_dict[slot] += 1
    
    # visualize the number of intent and slot for each label with two columns
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.bar(intent_label_lst, [math.log(intent_label_dict[label] + 1) for label in intent_label_lst], align='center')
    plt.xticks(intent_label_lst, intent_label_lst, rotation=90)
    plt.title('Intent')
    plt.subplot(1, 2, 2)
    plt.bar(slot_label_lst[:-2], [math.log(slot_label_dict[label] + 1) for label in slot_label_lst[:-2]], align='center')
    plt.xticks(slot_label_lst[:-2], slot_label_lst[:-2], rotation=90)
    plt.title('Slot')
    plt.savefig('BKAI/word-level/augment_train_val_plus/statistic.png')
    
    
if __name__ == '__main__':
    main()