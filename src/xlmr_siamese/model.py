import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLMRobertaModel, XLMRobertaForMaskedLM
from deeper_nlu.nn import MultiLayerPerceptron
from functools import partial
from .util import mean_encoded_seq_batch

class XLMRobertaSiamese(nn.Module):
    def __init__(self, model_name='xlm-roberta-large', pad_ix=1):
        super().__init__()
        self.mlm = XLMRobertaForMaskedLM.from_pretrained(model_name)
        self.pad_ix = pad_ix
    
    def forward(self, x):
        x = x[0]
        representations = self.mlm.roberta(x)[0]
        logits = self.mlm.lm_head(representations)
        # representations = mean_encoded_seq_batch(representations, x, ignore_index=self.pad_ix)
        representations = representations.mean(1)
        return logits, representations

class XLMRobertaForIcAndNer(nn.Module):
    def __init__(self, encoding_size, hidden_size, label_vocab_size, intent_vocab_size, n_hidden_layers=5, encoder_name='xlm-roberta-large', freeze_encoder=True):
        super().__init__()
        self.encoder = XLMRobertaModel.from_pretrained(encoder_name)
        if freeze_encoder:
            for p in self.encoder.parameters(): p.requires_grad = False
        self.lin_ner = MultiLayerPerceptron(encoding_size, hidden_size, label_vocab_size, n_hidden_layers=n_hidden_layers)
        self.lin_ic = MultiLayerPerceptron(encoding_size, hidden_size, intent_vocab_size, n_hidden_layers=n_hidden_layers)
    
    def forward(self, x, apply_softmax=False):
        encoded = self.encoder(x)[0]
        ner_out = self.lin_ner(encoded[:,1:,:])
        ic_out = self.lin_ic(encoded[:,0,:])
        if apply_softmax: ner_out, ic_out = map(partial(F.softmax, dim=-1), [ner_out, ic_out])
        return ic_out, ner_out