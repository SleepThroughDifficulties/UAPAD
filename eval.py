import argparse
import json
import logging
import os
import sys
import random
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report as cr

import torch
import csv
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AdamW, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
)
import model.utils as utils
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from pathlib import Path



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def parse_args():
    parser = argparse.ArgumentParser()
    # settings
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument("--dataset_name", default='glue', type=str)
    parser.add_argument("--task_name", default='sst2', type=str)
    parser.add_argument('--ckpt_dir', type=Path, default=Path('./saved_models/'))
    parser.add_argument('--num_labels', type=int, default=2)

    parser.add_argument('--prior_type', default='no_data',
                        help='Which kind of prior to use')
    parser.add_argument('--text_list', default='None',
                        help='In case of providing data priors,list of texts')
    parser.add_argument('--batch_size', default=25,
                        help='The batch size to use for training and testing')

    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--eval_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--weight_decay', default=1e-2, type=float)  # BERT default
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")  # BERT default
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Linear warmup over warmup_steps.")  # BERT default
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bias_correction', default=True)
    parser.add_argument('-f', '--force_overwrite', default=True)

    parser.add_argument('--attack_steps', default=300)
    parser.add_argument('--sample_size', default=32, help='每次更新用的训练集样本量')
    parser.add_argument('--attack_lr', default=0.05, type=float,
                        help='Step size of gradient ascent')
    parser.add_argument('--adv_init_mag', default=0, type=float,
                        help='随机初始化的范围值')
    parser.add_argument('--adv_max_norm', default=0.2, type=float,
                        help='adv_max_norm = 0 means unlimited')
    parser.add_argument('--adv_norm_type', default='l2', type=str,
                        help='norm type of the adversary')
    parser.add_argument('--perturb_num', default=20, type=int,
                        help='num of different deltas')

    args = parser.parse_args()
    if args.ckpt_dir is not None:
        os.makedirs(args.ckpt_dir, exist_ok=True)
    else:
        args.ckpt_dir = 'results'
    return args

def set_seed(seed: int):
    """Sets the relevant random seeds."""
    # seed = 111
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.

    From:
        https://github.com/uds-lsv/bert-stable-fine-tuning/blob/master/src/transformers/optimization.py
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def eval(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    avg_loss = utils.ExponentialMovingAverage()
    pred = []
    label = []
    with torch.no_grad():
        for model_inputs, labels in data_loader:
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            logits = model(**model_inputs).logits
            _, preds = logits.max(dim=-1)
            pred.append(preds.detach().cpu().numpy())
            label.append(labels.detach().cpu().numpy())
            loss = F.cross_entropy(logits, labels.squeeze(-1))
            avg_loss.update(loss.item())
            correct += (preds == labels.squeeze(-1)).sum().item()
            total += labels.size(0)
        accuracy = correct / (total + 1e-13)
        label = np.concatenate(label)
        pred = np.concatenate(pred)
        print(cr(label,pred))



    return accuracy, avg_loss.get_metric()

def evaluate(model, data_loader, device, delta):
    model.eval()
    correct = 0
    total = 0
    pred = []
    label = []
    avg_loss = utils.ExponentialMovingAverage()
    with torch.no_grad():
        for model_inputs, labels in data_loader:
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)

            word_embedding_layer = model.get_input_embeddings()
            input_ids = model_inputs['input_ids']
            del model_inputs['input_ids']  # new modified
            attention_mask = model_inputs['attention_mask']
            embedding_init = word_embedding_layer(input_ids)

            input_mask = attention_mask.to(embedding_init)
            # (0) forward
            mask_tensor = input_mask.unsqueeze(2)
            repeat_shape = mask_tensor.shape
            model_inputs['inputs_embeds'] = (delta.repeat(repeat_shape[0], repeat_shape[1], 1)
                                             * mask_tensor).to(torch.float32) + embedding_init  # new modified
            # batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
            logits = model(**model_inputs).logits
            _, preds = logits.max(dim=-1)
            loss = F.cross_entropy(logits, labels.squeeze(-1))
            avg_loss.update(loss.item())
            correct += (preds == labels.squeeze(-1)).sum().item()
            pred.append(preds.detach().cpu().numpy())
            label.append(labels.detach().cpu().numpy())
            total += labels.size(0)
        accuracy = correct / (total + 1e-13)
        label = np.concatenate(label)
        pred = np.concatenate(pred)
        print(cr(label, pred))

    return accuracy, avg_loss.get_metric()

def load_data(tokenizer, args):
    # dataloader
    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)
    # for training and dev
    train_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name, subset=args.task_name)

    if args.dataset_name == 'imdb' or args.dataset_name == 'ag_news':
        train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)

        test_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name,
                                                 subset=args.task_name, split='test')
        test_loader = DataLoader(test_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)
    elif args.task_name == 'mnli':
        train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
        dev_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name,
                                                 subset=args.task_name, split='validation_matched')
        dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)
        test_loader = dev_loader
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
        dev_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name,
                                                 subset=args.task_name, split='validation')
        dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)
        test_loader = dev_loader

    return train_dataset, train_loader, test_loader


def uap_attack(attack_path, args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = AutoConfig.from_pretrained(attack_path)
    model = AutoModelForSequenceClassification.from_pretrained(attack_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(attack_path)
    model.to(device)
    model.eval()

    train_dataset, train_loader, test_loader = load_data(tokenizer, args)

    dev_accuracy, dev_loss = eval(model, test_loader, device)
    logger.info(f'\nLoss: {dev_loss}, '
                f'Dev_Accuracy: {dev_accuracy}')

    with open(os.path.join('results', args.task_name, 'deltas0.json'), 'r') as d2:
        datas2 = json.load(d2)
    # d1 = torch.tensor(datas2[0])
    # d2 = torch.tensor(datas2[1])
    # delta = d1/torch.norm(d1)+d2/torch.norm(d2)
    # delta = -0.2*delta/torch.norm(delta)


    for delta in datas2:
        delta = torch.tensor(delta).to(device).view(1,1,-1)
        dev_accuracy, dev_loss = evaluate(model, test_loader, device, delta)
        logger.info(f'random_init， '
                        f'Loss: {dev_loss}, '
                        f'Dev_Accuracy: {dev_accuracy}, ')
        break



def main(args):
    set_seed(args.seed)
    if args.dataset_name == 'imdb' or args.dataset_name == 'ag_news':
        model_dir = Path(os.path.join(args.ckpt_dir, 'UAP_{}_{}_epochs{}_seed{}'
                                       .format(args.model_name, args.dataset_name,
                                                args.epochs, args.seed)))
    else:
        model_dir = Path(os.path.join(args.ckpt_dir, 'UAP_{}_{}-{}_epochs{}_seed{}'
                                       .format(args.model_name, args.dataset_name, args.task_name,
                                            args.epochs, args.seed)))
    if not model_dir.exists():
        logger.info(f'no such model_dir')
        return 0

    log_file = os.path.join(model_dir, 'attack_INFO.log')
    logger.addHandler(logging.FileHandler(log_file))

    uap_attack(model_dir, args)

if __name__ == '__main__':

    args = parse_args()

    level = logging.INFO
    logging.basicConfig(level=level)

    main(args)
