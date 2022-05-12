import copy
import json
import logging
import os
from random import shuffle, uniform
import numpy as np

import torch
from torch.utils.data import TensorDataset
from utils import get_intent_labels, get_slot_labels


logger = logging.getLogger(__name__)


class DataLoader(object):
    """
    DataLoader class
    """

    def __init__(self):
        self.dataset = {
            'train': {
                'data': [],
                'intent_label': [],
                'slot_label': [],
                'slot_words': [],
                'slot_train': [],
            },
            'val': {
                'data': [],
                'intent_label': [],
                'slot_label': [],
                'slot_word_val': [],
                'slot_val': []
                
            },
            'input-shape': 0,
            'output-shape': (0, 0)
        }
        
        self.intent_label_encoder = None
        self.slot_label_encoder = None

        self.dictionary = {
            'prefix': [
                (['đang', 'học', 'bài', 'vui', 'lòng'], ['O', 'O', 'O', 'O', 'O']),  
                ], 
            'suffix': [
                (['đê'], ['O']),
                (['chỉ', '1', 'chút', 'thôi','nhé'], ['O', 'O', 'O', 'O', 'O']),
                (['để', 'tôi', 'tắm'], ['O', 'O', 'O']),
                ],
            'smart.home.check.status': [
                (['toàn', 'bộ'], ['B-allall', 'I-allall']),
                (['báo', 'cáo'], ['B-commandcommand', 'I-commandcommand'])
                ],
            'smart.home.device.onoff': [
                
            ],
            
            'smart.home.set.color': [
                (['toàn', 'bộ'], ['B-allall', 'I-allall'])
            ],
            'smart.home.set.level': [
                (['toàn', 'bộ'], ['B-allall', 'I-allall']),
                (['ánh', 'sáng'], ['B-devicedevice', 'I-devicedevice']),
            ], 
            'smart.home.set.percentage': [
                (['toàn', 'bộ'], ['B-allall', 'I-allall']),
            ],
            'smart.home.increase.level': [
                (['toàn', 'bộ'], ['B-allall', 'I-allall']),
            ],
            'smart.home.increase.percentage': [
                (['toàn', 'bộ'], ['B-allall', 'I-allall']),
            ],
            'smart.home.decrease.level': [
                (['toàn', 'bộ'], ['B-allall', 'I-allall']),
            ],
            'smart.home.decrease.percentage': [
                (['toàn', 'bộ'], ['B-allall', 'I-allall']),
            ]
        }
        
        self.key_words = ['tăng', 'giảm', 'lên', 'xuống', 'mức', 'sáng', 'màu', 'thêm',
                    'cấp', 'bật', 'hạ', 'thay', 'đổi', 'số', 'chỉnh', 'điều']
        
    def make_dict(self, dataset):
        for sentence, intent, slots in zip(dataset['data'], dataset['intent_label'], dataset['slot_label']):
            words = sentence.split()
            slots = slots.split(' ')
            slots.append('.')
            idx = 0
            if intent != 'greeting':
                _n_prefix = 0
                for i, slot in enumerate(slots):
                    if slot == 'O' and words[i] not in self.key_words:
                        _n_prefix += 1
                    else:
                        break
                if _n_prefix < 6 and _n_prefix > 0 and (words[:_n_prefix], slots[:_n_prefix]) not in self.dictionary['prefix']:
                    self.dictionary['prefix'].append((words[:_n_prefix], slots[:_n_prefix]))
                
            while slots[idx] != '.':
                p_words = []
                p_slots = []
                num_p_words = 0
                if slots[idx][0] == 'B':
                    p_words.append(words[idx])
                    p_slots.append(slots[idx])
                    num_p_words += 1
                    idx += 1
                    while slots[idx][0] == 'I':
                        p_words.append(words[idx])
                        p_slots.append(slots[idx])
                        num_p_words += 1
                        idx += 1
                else:
                    p_words.append(words[idx])
                    p_slots.append(slots[idx])
                    num_p_words += 1
                    idx += 1
                
                if (intent, p_slots[0]) not in self.dictionary:
                    self.dictionary[(intent, p_slots[0])] = [(p_words, p_slots)]
                else:
                    self.dictionary[(intent, p_slots[0])].append((p_words, p_slots))
            if intent != 'greeting':
                _n_suffix = 0
                for i, slot in enumerate(slots[::-1][1:]):
                    if slot == 'O' and words[-i-1] not in self.key_words and not words[-i-1].isdigit():
                        _n_suffix += 1
                    else:
                        break
                if _n_suffix < 6 and _n_suffix > 0 and (words[-_n_suffix:], slots[-_n_suffix-1:-1]) not in self.dictionary['suffix']:
                    self.dictionary['suffix'].append((words[-_n_suffix:], slots[-_n_suffix-1:-1]))
    
    def augment(self, dataset, drop_rate=0.2, change_rate=0.2, K=4, merge=True):
        '''
        Augment data
        '''
        
        new_data = {
            'data': [],
            'intent_label': [],
            'slot_label': [],
        }
        
        existed = set()
         
        for sentence, intent, slots in zip(dataset['data'], dataset['intent_label'], dataset['slot_label']):
            words = sentence.split()
            slots = slots.split()
            slots.append('.')
            idx = 0
            p_word_list = []
            p_slot_list = []
            
            while slots[idx] != '.':
                p_word = []
                p_slot = []
                if slots[idx][0] == 'B':
                    p_word.append(words[idx])
                    p_slot.append(slots[idx])
                    idx += 1
                    while slots[idx][0] == 'I':
                        p_word.append(words[idx])
                        p_slot.append(slots[idx])
                        idx += 1
                else:
                    p_word.append(words[idx])
                    p_slot.append(slots[idx])
                    idx += 1
                
                p_word_list.append(p_word)
                p_slot_list.append(p_slot)
            
            for _ in range(K):
                p_words = []
                p_slots = []
                drop_p = uniform(0, 1)
                for p_word, p_slot in zip(p_word_list, p_slot_list):
                    if p_slot[0] != 'O' and uniform(0, 1) < change_rate:
                        rd_id = np.random.choice(len(self.dictionary[(intent, p_slot[0])]))
                        p_word, p_slot = self.dictionary[(intent, p_slot[0])][rd_id]
                    if drop_p < drop_rate and p_word[0] not in self.key_words:
                        if p_slot[0] == 'O':
                            continue
                    p_words.append(' '.join(p_word))
                    p_slots.append(' '.join(p_slot))
                
                if len(p_words) > 1:
                    if ((p_words[0].split(' ')[0] in self.key_words) or p_slots[0] != 'O') and uniform(0, 1) < 0.5 and len(self.dictionary['prefix']) > 0:
                        # add random prefix
                        rd_id = np.random.choice(len(self.dictionary['prefix']))
                        p_word, p_slot = self.dictionary['prefix'][rd_id]
                        p_words.insert(0, ' '.join(p_word))
                        p_slots.insert(0, ' '.join(p_slot))
                    
                    if p_slots[-1] != 'O' and uniform(0, 1) < 0.5 and len(self.dictionary['suffix']) > 0:
                        # add random suffix
                        rd_id = np.random.choice(len(self.dictionary['suffix']))
                        p_word, p_slot = self.dictionary['suffix'][rd_id]
                        p_words.append(' '.join(p_word))
                        p_slots.append(' '.join(p_slot))
                    
                    new_segmented_sentence = ' '.join(p_words)
                    new_segmented_slots = ' '.join(p_slots)
                    hashing_new_sentence = hash(new_segmented_sentence)
                    if hashing_new_sentence not in existed:
                        existed.add(hashing_new_sentence)
                        new_data['data'].append(new_segmented_sentence)
                        new_data['intent_label'].append(intent)
                        new_data['slot_label'].append(new_segmented_slots)
                        
        # merge data
        if merge:
            for sentence, intent, slots in zip(dataset['data'], dataset['intent_label'], dataset['slot_label']):
                new_data['data'].append(sentence)
                new_data['intent_label'].append(intent)
                new_data['slot_label'].append(slots)
                
        # shuffle data
        ids = list(range(len(new_data['data'])))
        np.random.shuffle(ids)
        new_data['data'] = [new_data['data'][i] for i in ids]
        new_data['intent_label'] = [new_data['intent_label'][i] for i in ids]
        new_data['slot_label'] = [new_data['slot_label'][i] for i in ids]
        
        return new_data

    def dump(self, dataset=None, path=None):
        '''
        Dump data to pkl format
        '''
        if not os.path.exists(path):
            os.makedirs(path)
            
        seqin_file = os.path.join(path, 'seq.in')
        seqout_file = os.path.join(path, 'seq.out')
        label_file = os.path.join(path, 'label')
        with open(seqin_file, 'w') as f:
            for sentence in dataset['data']:
                f.write(sentence + '\n')
        with open(label_file, 'w') as f:
            for intent in dataset['intent_label']:
                f.write(intent + '\n')
        with open(seqout_file, 'w') as f:
            for slot in dataset['slot_label']:
                f.write(slot + '\n')
        
    def make(self, dataset_path=None, trainset_path=None, train_intent_label=None, train_slot_label=None,
             valset_path=None, val_intent_label=None, val_slot_label=None,
             intent_label_file=None, slot_label_file=None):
        
        ''' Load data '''
        trainset, train_intel_label, train_slot_label = \
            load_data(dataset_path, trainset_path,
                        train_intent_label, train_slot_label)
        
        valset, val_intel_label, val_slot_label = \
            load_data(dataset_path, valset_path,
                        val_intent_label, val_slot_label)
        self.dataset['train']['data'] = trainset
        self.dataset['train']['intent_label'] = train_intel_label
        self.dataset['train']['slot_label'] = train_slot_label
        
        self.dataset['val']['data'] = valset
        self.dataset['val']['intent_label'] = val_intel_label
        self.dataset['val']['slot_label'] = val_slot_label
        

        intent_label_list = []
        slot_label_list = []

        with open(intent_label_file, 'r') as f:
            for line in f:
                intent_label_list.append(line.strip())
        with open(slot_label_file, 'r') as f:
            for line in f:
                slot_label_list.append(line.strip())
            
class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        intent_label: (Optional) string. The intent label of the example.
        slot_labels: (Optional) list. The slot labels of the example.
    """

    def __init__(self, guid, words, intent_label=None, slot_labels=None):
        self.guid = guid
        self.words = words
        self.intent_label = intent_label
        self.slot_labels = slot_labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, intent_label_id, slot_labels_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.intent_label_id = intent_label_id
        self.slot_labels_ids = slot_labels_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class JointProcessor(object):
    """Processor for the JointBERT data set """

    def __init__(self, args):
        self.args = args
        self.intent_labels = get_intent_labels(args)
        self.slot_labels = get_slot_labels(args)
        if args.use_filter:
            self.input_text_file = "filtered_seq.in"
            self.intent_label_file = "filtered_label"
            self.slot_labels_file = "filtered_seq.out"
        else:
            self.input_text_file = "seq.in"
            self.intent_label_file = "label"
            self.slot_labels_file = "seq.out"
            
    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, texts, intents, slots, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, (text, intent, slot) in enumerate(zip(texts, intents, slots)):
            guid = "%s-%s" % (set_type, i)
            # 1. input_text
            words = text.split()  # Some are spaced twice
            # 2. intent
            intent_label = (
                self.intent_labels.index(intent) if intent in self.intent_labels else self.intent_labels.index("UNK")
            )
            # 3. slot
            slot_labels = []
            for s in slot.split():
                slot_labels.append(
                    self.slot_labels.index(s) if s in self.slot_labels else self.slot_labels.index("UNK")
                )

            assert len(words) == len(slot_labels)
            examples.append(InputExample(guid=guid, words=words, intent_label=intent_label, slot_labels=slot_labels))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        data_path = os.path.join(self.args.data_dir, self.args.token_level, mode)
        logger.info("LOOKING AT {}".format(data_path))
        return self._create_examples(
            texts=self._read_file(os.path.join(data_path, self.input_text_file)),
            intents=self._read_file(os.path.join(data_path, self.intent_label_file)),
            slots=self._read_file(os.path.join(data_path, self.slot_labels_file)),
            set_type=mode,
        )


processors = {"syllable-level": JointProcessor, "word-level": JointProcessor}


def convert_examples_to_features(
    examples,
    max_seq_len,
    tokenizer,
    pad_token_label_id=-100,
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

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # Tokenize word by word (for NER)
        tokens = []
        slot_labels_ids = []
        for word, slot_label in zip(example.words, example.slot_labels):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_labels_ids.extend([int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[: (max_seq_len - special_tokens_count)]
            slot_labels_ids = slot_labels_ids[: (max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        slot_labels_ids += [pad_token_label_id]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        slot_labels_ids = [pad_token_label_id] + slot_labels_ids
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_labels_ids = slot_labels_ids + ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len
        )
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(
            len(token_type_ids), max_seq_len
        )
        assert len(slot_labels_ids) == max_seq_len, "Error with slot labels length {} vs {}".format(
            len(slot_labels_ids), max_seq_len
        )

        intent_label_id = int(example.intent_label)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("intent_label: %s (id = %d)" % (example.intent_label, intent_label_id))
            logger.info("slot_labels: %s" % " ".join([str(x) for x in slot_labels_ids]))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                intent_label_id=intent_label_id,
                slot_labels_ids=slot_labels_ids,
            )
        )

    return features


def load_and_cache_examples(args, tokenizer, mode):
    processor = processors[args.token_level](args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            mode, args.token_level, list(filter(None, args.model_name_or_path.split("/"))).pop(), args.max_seq_len
        ),
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # Load data features from dataset file
        if mode in ["train", "augment_train", "augment_train_val", "augment_val", "augment_train_val_plus",
                    "train_val", "train_val_test", "dev"]:
            examples = processor.get_examples(mode)
        elif mode == "test":
            examples = processor.get_examples(mode)
        else:
            raise Exception("For mode {}, Only train, dev, test is available".format(mode))

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        pad_token_label_id = args.ignore_index
        features = convert_examples_to_features(
            examples, args.max_seq_len, tokenizer, pad_token_label_id=pad_token_label_id
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_intent_label_ids = torch.tensor([f.intent_label_id for f in features], dtype=torch.long)
    all_slot_labels_ids = torch.tensor([f.slot_labels_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids, all_attention_mask, all_token_type_ids, all_intent_label_ids, all_slot_labels_ids
    )
    return dataset


def load_data(data_path=None, data_file_name=None, data_intent_label=None,
            data_slot_label=None):
    """
    Load data from the dataset.
    """
    
    _data = []
    _intent_label = []
    _slot_label = []
    
    # Read lines from the file, split by \n
    if data_file_name:
            data_file_path = os.path.join(data_path, data_file_name)
            if os.path.exists(data_file_path):
                with open(data_file_path, 'r') as f:
                    _data = f.read().split('\n')
    if data_intent_label:
        data_intent_label_path = os.path.join(data_path, data_intent_label)
        if os.path.exists(data_intent_label_path):
            with open(data_intent_label_path, 'r') as f:
                _intent_label = f.read().split('\n')
    
    if data_slot_label:
        data_slot_label_path = os.path.join(data_path, data_slot_label)
        if os.path.exists(data_slot_label_path):
            with open(data_slot_label_path, 'r') as f:
                _slot_label = f.read().split('\n')
    
    data = []
    intent_label = []
    slot_label = []
    
    # split each line by ' '
    data = [line.lower() for line in _data if len(line) != 0]
    intent_label = [line for line in _intent_label if len(line) != 0]
    slot_label = [line for line in _slot_label if len(line) != 0]

    return data, intent_label, slot_label