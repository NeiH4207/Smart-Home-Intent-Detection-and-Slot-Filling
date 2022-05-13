import torch
import torch.nn as nn
from torchcrf import CRF
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel
import torch.nn.functional as F
from .module import IntentClassifier, SlotClassifier
import pandas as pd
from torch import optim

# Bi-model 
class slot_enc(nn.Module):
    def __init__(self, embedding_size, lstm_hidden_size, dropout_rate=0.5, device='cpu'):
        super(slot_enc, self).__init__()
        self.dropout_rate = dropout_rate
        self.device = device
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=lstm_hidden_size, num_layers=2,\
                            bidirectional= True, batch_first=True) #, dropout=DROPOUT)

    def forward(self, x):    
        x, _ = self.lstm(x)
        x = F.dropout(x, self.dropout_rate)
        return x 


class slot_dec(nn.Module):
    def __init__(self, lstm_hidden_size, label_size=1, dropout_rate=0.5, device='cpu'):
        super(slot_dec, self).__init__()
        self.dropout_rate = dropout_rate
        self.device = device
        self.lstm = nn.LSTM(input_size=lstm_hidden_size*5, hidden_size=lstm_hidden_size, num_layers=1)
        self.fc = nn.Linear(lstm_hidden_size, label_size)
        self.hidden_size = lstm_hidden_size

    def forward(self, x, hi):
        batch = x.size(0)
        length = x.size(1)
        dec_init_out = torch.zeros(batch, 1, self.hidden_size).to(self.device)
        hidden_state = (torch.zeros(1, 1, self.hidden_size).to(self.device), \
                        torch.zeros(1, 1, self.hidden_size).to(self.device))
        x = torch.cat((x, hi), dim=-1)

        x = x.transpose(1, 0)  # 50 x batch x feature_size
        x = F.dropout(x, self.dropout_rate)
        all_out = []
        for i in range(length):
            if i == 0:
                out, hidden_state = self.lstm(torch.cat((x[i].unsqueeze(1), dec_init_out), dim=-1), hidden_state)
            else:
                out, hidden_state = self.lstm(torch.cat((x[i].unsqueeze(1), out), dim=-1), hidden_state)
            all_out.append(out)
        output = torch.cat(all_out, dim=1) # 50 x batch x feature_size
        x = F.dropout(x, self.dropout_rate)
        res = self.fc(output)
        return res 

class intent_enc(nn.Module):
    def __init__(self, embedding_size, lstm_hidden_size, dropout_rate=0.3, device='cpu'):
        super(intent_enc, self).__init__()
        self.dropout_rate = dropout_rate
        self.device = device
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size= lstm_hidden_size, num_layers=2,\
                            bidirectional= True, batch_first=True, dropout=self.dropout_rate)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = F.dropout(x, self.dropout_rate)
        return x

class intent_dec(nn.Module):
    def __init__(self, lstm_hidden_size, label_size=1, dropout_rate=0.3, device='cpu'):
        super(intent_dec, self).__init__()
        self.device = device
        self.dropout_rate = dropout_rate
        self.lstm = nn.LSTM(input_size=lstm_hidden_size*4, hidden_size=lstm_hidden_size, batch_first=True, num_layers=1)#, dropout=DROPOUT)
        self.fc = nn.Linear(lstm_hidden_size, label_size)
        
    def forward(self, x, hs, real_len):
        batch = x.size()[0]
        real_len = torch.tensor(real_len).to(self.device)
        x = torch.cat((x, hs), dim=-1)
        x = F.dropout(x, self.dropout_rate)
        x, _ = self.lstm(x)
        x = F.dropout(x, self.dropout_rate)

        index = torch.arange(batch).long().to(self.device)
        state = x[index, real_len-1, :]
        
        res = self.fc(state.squeeze())
        return res

class Intent(nn.Module):
    def __init__(self, embedding_size, lstm_hidden_size, batch_size, max_len, device='cpu', label_size=1, dropout_rate=0.3):
        super(Intent, self).__init__()
        self.enc = intent_enc(embedding_size, lstm_hidden_size, dropout_rate, device).to(device)
        self.dec = intent_dec(lstm_hidden_size, label_size, dropout_rate, device).to(device)
        self.share_memory = torch.zeros(batch_size, max_len, lstm_hidden_size * 2).to(device)
    

class Slot(nn.Module):
    def __init__(self, embedding_size, lstm_hidden_size, batch_size, max_len, device='cpu', label_size=1, dropout_rate=0.3):
        super(Slot, self).__init__()
        self.enc = slot_enc(embedding_size, lstm_hidden_size, dropout_rate, device).to(device)
        self.dec = slot_dec(lstm_hidden_size, label_size, dropout_rate, device).to(device)
        self.share_memory = torch.zeros(batch_size, max_len, lstm_hidden_size * 2).to(device)


def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)

        vector = vector + (mask + 1e-45).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)

def make_mask(real_len, max_len=None, label_size=None):
    batch_size = len(real_len)
    mask = torch.zeros(batch_size, max_len, label_size)
    for index, item in enumerate(real_len):
        mask[index, :item, :] = 1.0
    return mask

class BiModel(RobertaPreTrainedModel):
    def __init__(self, config, args, intent_label_lst, slot_label_lst):
        super(BiModel, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.roberta = RobertaModel(config)  # Load pretrained phobert    
        device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'

        self.slot_classifier = Slot(config.hidden_size, config.hidden_size, args.train_batch_size, args.max_seq_len, device, label_size=self.num_slot_labels)
        self.intent_classifier = Intent(config.hidden_size, config.hidden_size, args.train_batch_size, args.max_seq_len, device, label_size=self.num_intent_labels)
        
        self.rule_matrix = pd.read_csv(args.rule_file, sep=',', header=None).to_numpy()
        self.rule_matrix = torch.FloatTensor(self.rule_matrix)
        self.alpha = 0.5
        
        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)
        # self.slot_optimizer = optim.Adam(self.slot_classifier.parameters(), lr=args.learning_rate)       # optim.Adamax
        # self.intent_optimizer = optim.Adam(self.intent_classifier.parameters(), lr=args.learning_rate)   # optim.Adamax
        
    def forward(self, input_ids, attention_mask, token_type_ids, 
                intent_label_ids, slot_labels_ids, real_lens):
		# Calculate compute graph
        outputs = self.roberta(
            input_ids, attention_mask=attention_mask, 
        )
        mask = make_mask(real_lens,self. args.max_seq_len, self.num_slot_labels).to(self.roberta.device)
        embed_vecs = outputs[0]
        hs = self.slot_classifier.enc(embed_vecs)
        self.slot_classifier.share_memory = hs.clone()
        
        hi = self.intent_classifier.enc(embed_vecs)
        self.intent_classifier.share_memory = hi.clone()
        
        slot_logits = self.slot_classifier.dec(hs, self.intent_classifier.share_memory.detach())
        log_slot_logits = masked_log_softmax(slot_logits, mask, dim=-1)
        
        # Asynchronous training
        intent_logits = self.intent_classifier.dec(hi, self.slot_classifier.share_memory.detach(), real_lens)
        log_intent_logits = F.log_softmax(intent_logits, dim=-1)
        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(
                    intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1)
                )
            total_loss += self.args.intent_loss_coef * intent_loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            if self.args.use_crf:
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction="mean")
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += (1 - self.args.intent_loss_coef) * slot_loss

        if self.args.use_rule_based:
            slot_probs = torch.exp(slot_logits).permute(1, 0, 2)
            num_words = slot_probs.shape[0]
            for T in range(num_words):
                if T == 0:
                    continue
                argmax_idx = torch.argmax(slot_probs[T-1], dim=-1)
                onehot_vec = F.one_hot(argmax_idx, self.num_slot_labels).float()
                slot_probs[T] *= torch.matmul(onehot_vec, self.rule_matrix.to(self.device))
                slot_probs[T] /= torch.sum(slot_probs[T], dim=1, keepdim=True)
            slot_logits = torch.log(slot_probs.permute(1, 0, 2))
            
        outputs = ((intent_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits
