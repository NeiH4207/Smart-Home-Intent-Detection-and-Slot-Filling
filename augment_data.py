"""
Created on Mon Mar 11 16:00:47 2022
@author: hien
"""
from __future__ import division
import argparse
from copy import deepcopy
import logging
import os
from src.data_loader import DataLoader
log = logging.getLogger(__name__)

from src.data_helper import *

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
    
    args = parser.parse_args()
    args.intent_label_file = os.path.join(args.dataset_path, args.intent_label_file)
    args.slot_label_file = os.path.join(args.dataset_path, args.slot_label_file)
    return args


def main():
    args = parse_args()
    data_loader = DataLoader()
    
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
                                                  merge=True, drop_rate=0.5, change_rate=1.0, K=10)
    agumented_val_dataset = data_loader.augment(data_loader.dataset['val'], 
                                                merge=True, drop_rate=0.5, change_rate=1.0, K=10)
    agumented_train_val_dataset = deepcopy(agumented_train_dataset)
    
    for sentence, intent, slots in zip(agumented_val_dataset['data'], 
                                       agumented_val_dataset['intent_label'], 
                                       agumented_val_dataset['slot_label']):
        agumented_train_val_dataset['data'].append(sentence)
        agumented_train_val_dataset['intent_label'].append(intent)
        agumented_train_val_dataset['slot_label'].append(slots)
    
    data_loader.dump(path='BKAI/word-level/augment_train',
                     dataset=agumented_train_dataset)
    data_loader.dump(path='BKAI/word-level/augment_train_val_plus',
                     dataset=agumented_train_val_dataset)
    
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
                    
    
if __name__ == '__main__':
    main()