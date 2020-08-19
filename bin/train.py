import argparse
from transformers import XLMRobertaTokenizer, XLMRobertaConfig
from deeper_nlu.data import TabularDataset, CategoryProcessor, TokenizeProcessor, NumericalizeProcessor
from deeper_nlu.train import Learner, CombinedLoss, AvgStatsCallback, CudaCallback, SaveModel, EarlyStop
from deeper_nlu.metric import combined_accuracy, class_accuracy, seq_accuracy
import torch
import torch.nn as nn
from torch.optim import Adam
from pathlib import Path
from datetime import datetime
import json
from model import XLMRobertaForIcAndNer
from functools import partial

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--hidden-size', type=int, default=350)
    parser.add_argument('--hidden-layers', type=int, default=5)
    parser.add_argument('--encoder-name', type=str, default='xlm-roberta-large')
    return parser.parse_args()

args = parse_args()

root = Path(args.output) if args.output else Path.cwd()
path = root/datetime.now().strftime('%Y%m%d')/'finetune'
path.mkdir(parents=True, exist_ok=True)

with open(path/'hp.json', 'w') as f: json.dump(args.__dict__, f, indent=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_count = torch.cuda.device_count() or 1

annotation_to_labels = lambda anno: ' '.join([item.split('|')[1] for item in anno.split()])
domain_intent_join = lambda domain, intent: '_'.join([domain, intent]) if domain and intent else ''

tfms = {
    'text': lambda d: ' '.join([item.split('|')[0] for item in d['annotation'].split()]),
    'intent': lambda d: domain_intent_join(d['domain'], d['intent']),
    'labels': lambda d: annotation_to_labels(d['annotation']),
}

tokenizer = XLMRobertaTokenizer.from_pretrained(args.encoder_name)

proc_x = {
    'text': partial(tokenizer.encode, add_special_tokens=True)
}

proc_y = {
    'intent': CategoryProcessor(min_freq=0),
    'labels': [TokenizeProcessor(), NumericalizeProcessor(min_freq=0)]
}

data = TabularDataset \
    .from_csv(args.input, names=['domain', 'intent', 'annotation', 'cust_id', 'utt_id'], tfms=tfms) \
    .split(stratify_by='domain') \
    .label(proc_x=proc_x, proc_y=proc_y) \
    .to_databunch(bs=args.batch_size*device_count)

with open(path/'data.json', 'w') as o: json.dump(data.to_serializable(), o)

proc_x['text'][1].save(path/'word_vocab.json')
proc_x['labels'][1].save(path/'label_vocab.json')
proc_x['intent'].save(path/'intent_vocab.json')

encoding_size = XLMRobertaConfig.from_pretrained(args.encoder_name).hidden_size
label_vocab_size = len(proc_y['labels'][1].vocab)
intent_vocab_size = len(proc_y['intent'].vocab)

model = XLMRobertaForIcAndNer(encoding_size, args.hidden_size, label_vocab_size, intent_vocab_size, n_hidden_layers=args.hidden_layers, encoder_name=args.encoder_name)

if device_count > 1: model = nn.DataParallel(model)

loss_func = CombinedLoss()

cbfs = [
    partial(AvgStatsCallback, [combined_accuracy, class_accuracy, seq_accuracy], path=path/'metrics.tsv'),
    partial(CudaCallback, device),
    partial(SaveModel, path/'model.pth'),
    EarlyStop
]
opt = partial(Adam, lr=args.learning_rate, weight_decay=args.weight_decay)
learn = Learner(model, data, loss_func, lr=args.learning_rate, cb_funcs=cbfs, opt_func=opt)

print('Training model on {device_count} {device_type}(s)'.format(device_count=device_count, device_type=device))
learn.fit(50)