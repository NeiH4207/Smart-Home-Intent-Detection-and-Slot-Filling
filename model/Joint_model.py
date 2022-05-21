import torch
import torch.nn as nn
from torchcrf import CRF
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel
import torch.nn.functional as F
from .module import IntentClassifier, SlotClassifier
import pandas as pd

class SelfAttention(nn.Module):

    def __init__(self, input_dim, output_dim, dropout_rate):
        super(SelfAttention, self).__init__()

        self._k_matrix = nn.Linear(input_dim, output_dim)
        self._v_matrix = nn.Linear(input_dim, output_dim)
        self._q_matrix = nn.Linear(input_dim, output_dim)
        self._dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, input_x):
        k_x = self._k_matrix(input_x)
        v_x = self._v_matrix(input_x)
        q_x = self._q_matrix(input_x)

        drop_kx = self._dropout_layer(k_x)
        drop_vx = self._dropout_layer(v_x)
        drop_qx = self._dropout_layer(q_x)

        alpha = F.softmax(torch.matmul(drop_qx.transpose(-2, -1), drop_kx), dim=-1)
        return torch.matmul(drop_vx, alpha)

class Embedding(nn.Module):
    def __init__(self, vocab_size=64000, embedding_size=None):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.embedding.weight.data.uniform_(0, 0.1)

    def forward(self, input):
        return self.embedding(input)

class JointGLU(RobertaPreTrainedModel):
    def __init__(self, config, args, intent_label_lst, slot_label_lst):
        super(JointGLU, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.roberta = RobertaModel(config)  # Load pretrained phobert

        self._embedding_layer = Embedding(64000, config.hidden_size)
        
        self.emb_fc = nn.Linear(config.hidden_size * 2, config.hidden_size)
        
        self._e_attention = SelfAttention(
            config.hidden_size, args.attention_embedding_size, args.dropout_rate
        )
        self._lstm_layer = nn.LSTM(
            input_size=config.hidden_size+args.attention_embedding_size,
            hidden_size=args.attention_embedding_size // 2,
            batch_first=True,
            num_layers=1,
            bidirectional=True,
            dropout=args.dropout_rate
        )
        
        self._d_attention = SelfAttention(
            args.attention_embedding_size, args.attention_embedding_size, args.dropout_rate
        )
        
        self.intent_classifier = IntentClassifier(config.hidden_size + 2 * args.attention_embedding_size, 
                                                  self.num_intent_labels, args.dropout_rate)
        
        self._intent_gate_linear = nn.Linear(
            args.attention_embedding_size + self.num_intent_labels, args.attention_embedding_size
        )
        
        self._lstm_slot_layer = nn.LSTM(
            input_size=args.attention_embedding_size * 2,
            hidden_size=config.hidden_size // 2,
            batch_first=True,
            num_layers=1,
            bidirectional=True,
            dropout=args.dropout_rate
        )
        
        self.slot_classifier = SlotClassifier(
            config.hidden_size,
            self.num_intent_labels,
            self.num_slot_labels,
            self.args.use_intent_context_concat,
            self.args.use_intent_context_attention,
            self.args.max_seq_len,
            self.args.attention_embedding_size,
            args.dropout_rate,
        )

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)
        if args.use_rule_based:
            self.rule_matrix = pd.read_csv(args.rule_file, sep=',', header=None).to_numpy()
            self.rule_matrix = torch.FloatTensor(self.rule_matrix)
        self.alpha = 0.5
        
    def forward(self, input_ids, attention_mask, token_type_ids, 
                intent_label_ids, slot_labels_ids, real_lens):
        outputs = self.roberta(
            input_ids, attention_mask=attention_mask, 
        )
        attention_x = self._e_attention(outputs[0])
        emb_attn_x = torch.cat([outputs[0], attention_x], dim=-1)
        lstm_hidden, (hn, _) = self._lstm_layer(emb_attn_x)
        hn = torch.cat([hn[0], hn[1]], dim=-1)
        pool_hidden = torch.max(lstm_hidden, dim=1, keepdim=True).values
        pool_hidden = torch.cat([hn, pool_hidden.squeeze(1), outputs.pooler_output], dim=-1).unsqueeze(1)
        linear_intent = self.intent_classifier(pool_hidden)
        intent_logits = F.log_softmax(linear_intent.squeeze(1), dim=-1)
        # SLU
        rep_intent = torch.cat([linear_intent] * input_ids.size(1), dim=1)
        attn_hidden = self._d_attention(lstm_hidden)
        com_hidden = torch.cat([rep_intent, attn_hidden], dim=-1)
        lin_hidden = self._intent_gate_linear(com_hidden)
        gated_hidden = lin_hidden * lstm_hidden
        sequence_output = torch.cat((lstm_hidden, gated_hidden), dim=-1)
        sequence_output = self._lstm_slot_layer(sequence_output)[0]
        if not self.args.use_attention_mask:
            tmp_attention_mask = None
        else:
            tmp_attention_mask = attention_mask

        if self.args.embedding_type == "hard":
            hard_intent_logits = torch.zeros(intent_logits.shape)
            for i, sample in enumerate(intent_logits):
                max_idx = torch.argmax(sample)
                hard_intent_logits[i][max_idx] = 1
            slot_logits = self.slot_classifier(sequence_output, hard_intent_logits, tmp_attention_mask)
        else:
            slot_logits = self.slot_classifier(sequence_output, intent_logits, tmp_attention_mask)
            slot_logits = F.log_softmax(slot_logits, dim=-1)   
            
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
            slot_probs = torch.exp(slot_logits.permute(1, 0, 2))
            num_words = slot_probs.shape[0]
            for T in range(num_words):
                if T == 0:
                    continue
                if T < num_words - 1:
                    # make PAD to be 0
                    for i in range(slot_probs[T].shape[0]):
                        slot_probs[T][i][self.args.ignore_index] = 0
                
                argmax_idx = torch.argmax(slot_probs[T-1], dim=-1)
                onehot_vec = F.one_hot(argmax_idx, self.num_slot_labels).float()
                slot_probs[T] *= torch.matmul(onehot_vec, self.rule_matrix.to(self.device))
                slot_probs[T] /= torch.sum(slot_probs[T], dim=1, keepdim=True)
            slot_logits = torch.log(slot_probs.permute(1, 0, 2))
            
        outputs = ((intent_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits

    def get_features(self, input_ids, attention_mask, token_type_ids):
        outputs = self.roberta(
            input_ids, attention_mask=attention_mask, 
        )
        attention_x = self._e_attention(outputs[0])
        emb_attn_x = torch.cat([outputs[0], attention_x], dim=-1)
        lstm_hidden, (hn, _) = self._lstm_layer(emb_attn_x)
        hn = torch.cat([hn[0], hn[1]], dim=-1)
        pool_hidden = torch.max(lstm_hidden, dim=1, keepdim=True).values
        pool_hidden = torch.cat([hn, pool_hidden.squeeze(1), outputs.pooler_output], dim=-1)
        return pool_hidden
    
    def predict(self, input_ids, attention_mask, token_type_ids):
        outputs = self.roberta(
            input_ids, attention_mask=attention_mask, 
        )
        attention_x = self._e_attention(outputs[0])
        emb_attn_x = torch.cat([outputs[0], attention_x], dim=-1)
        lstm_hidden, (hn, _) = self._lstm_layer(emb_attn_x)
        hn = torch.cat([hn[0], hn[1]], dim=-1)
        pool_hidden = torch.max(lstm_hidden, dim=1, keepdim=True).values
        pool_hidden = torch.cat([hn, pool_hidden.squeeze(1), outputs.pooler_output], dim=-1).unsqueeze(1)
        linear_intent = self.intent_classifier(pool_hidden)
        intent_logits = F.log_softmax(linear_intent.squeeze(1), dim=-1)
        # SLU
        rep_intent = torch.cat([linear_intent] * input_ids.size(1), dim=1)
        attn_hidden = self._d_attention(lstm_hidden)
        com_hidden = torch.cat([rep_intent, attn_hidden], dim=-1)
        lin_hidden = self._intent_gate_linear(com_hidden)
        gated_hidden = lin_hidden * lstm_hidden
        sequence_output = torch.cat((lstm_hidden, gated_hidden), dim=-1)
        sequence_output = self._lstm_slot_layer(sequence_output)[0]
        if not self.args.use_attention_mask:
            tmp_attention_mask = None
        else:
            tmp_attention_mask = attention_mask

        if self.args.embedding_type == "hard":
            hard_intent_logits = torch.zeros(intent_logits.shape)
            for i, sample in enumerate(intent_logits):
                max_idx = torch.argmax(sample)
                hard_intent_logits[i][max_idx] = 1
            slot_logits = self.slot_classifier(sequence_output, hard_intent_logits, tmp_attention_mask)
        else:
            slot_logits = self.slot_classifier(sequence_output, intent_logits, tmp_attention_mask)
            slot_logits = F.log_softmax(slot_logits, dim=-1)   
            

        if self.args.use_rule_based:
            slot_probs = torch.exp(slot_logits.permute(1, 0, 2))
            num_words = slot_probs.shape[0]
            for T in range(num_words):
                if T == 0:
                    continue
                if T < num_words - 1:
                    # make PAD to be 0
                    for i in range(slot_probs[T].shape[0]):
                        slot_probs[T][i][self.args.ignore_index] = 0
                
                argmax_idx = torch.argmax(slot_probs[T-1], dim=-1)
                onehot_vec = F.one_hot(argmax_idx, self.num_slot_labels).float()
                slot_probs[T] *= torch.matmul(onehot_vec, self.rule_matrix.to(self.device))
                slot_probs[T] /= torch.sum(slot_probs[T], dim=1, keepdim=True)
            slot_logits = torch.log(slot_probs.permute(1, 0, 2))
        
        outputs = (intent_logits, slot_logits)
        return outputs