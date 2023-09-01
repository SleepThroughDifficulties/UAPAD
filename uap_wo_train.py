import argparse
import json
import logging
import os

def parse_args():
    parser = argparse.ArgumentParser()
    # settings
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument("--dataset_name", default='imdb', type=str)
    parser.add_argument("--task_name", default='none', type=str)
    parser.add_argument('--ckpt_dir', type=str, default='/workspace/saved_models/finetune/imdb/none/finetune_bert-base-uncased_imdb_lr2e-05_epochs3/epoch2')
    parser.add_argument('--num_labels', type=int, default=2)

    parser.add_argument('--prior_type', default='no_data',
                        help='Which kind of prior to use')
    parser.add_argument('--text_list', default='None',
                        help='In case of providing data priors,list of texts')
    parser.add_argument('--batch_size', default=25,
                        help='The batch size to use for training and testing')

    parser.add_argument('--max_seq_length', type=int, default=256)
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

    parser.add_argument('--attack_steps', default=300, type=int)
    parser.add_argument('--sample_size', default=32, help='每次更新用的训练集样本量')
    parser.add_argument('--attack_lr', default=0.05, type=float,
                        help='Step size of gradient ascent')
    parser.add_argument('--adv_init_mag', default=0.2, type=float,
                        help='随机初始化的范围值')
    parser.add_argument('--adv_max_norm', default=0.2, type=float,
                        help='adv_max_norm = 0 means unlimited')
    parser.add_argument('--adv_norm_type', default='l2', type=str,
                        help='norm type of the adversary')
    parser.add_argument('--perturb_num', default=10, type=int,
                        help='num of different deltas')
    parser.add_argument('--class_bound', default=0.5, type=float,
                        help='分类的偏差超过这个数值才是好的UAP')

    parser.add_argument('--generate_step', default=20, type=int,
                        help='生成虚拟数据的迭代次数')
    parser.add_argument('--cuda', default=1, type=int)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.cuda}"

    if args.ckpt_dir is not None:
        os.makedirs(args.ckpt_dir, exist_ok=True)
    else:
        args.ckpt_dir = '.'
    return args

args = parse_args()


import sys
import random
import numpy as np
import pandas as pd

import torch
import csv
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AdamW, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
)
import detect_utils.utils as utils
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from pathlib import Path



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def set_seed(seed: int):
    """Sets the relevant random seeds."""
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
    with torch.no_grad():
        for model_inputs, labels in data_loader:
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            logits = model(**model_inputs).logits
            _, preds = logits.max(dim=-1)
            loss = F.cross_entropy(logits, labels.squeeze(-1))
            avg_loss.update(loss.item())
            correct += (preds == labels.squeeze(-1)).sum().item()
            total += labels.size(0)
        accuracy = correct / (total + 1e-13)
    return accuracy, avg_loss.get_metric()

def evaluate(model, data_loader, device, delta, class_bound):
    model.eval()
    correct = 0
    total = 0
    delta_type = None
    pred_label = 0
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

            logits = model(**model_inputs).logits
            _, preds = logits.max(dim=-1)
            loss = F.cross_entropy(logits, labels.squeeze(-1))
            avg_loss.update(loss.item())
            pred_label += (preds == 1).sum().item()
            correct += (preds == labels.squeeze(-1)).sum().item()
            total += labels.size(0)
        accuracy = correct / (total + 1e-13)
        if pred_label/(total + 1e-13) > class_bound:
            delta_type = 1
        elif pred_label/(total + 1e-13) < 1-class_bound:
            delta_type = 0
    return accuracy, avg_loss.get_metric(), delta_type

def load_data(tokenizer, args):
    # dataloader
    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)

    if args.dataset_name == 'imdb' or args.dataset_name == 'ag_news':
        train_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name)
        train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)

        test_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name, split='test')
        test_loader = DataLoader(test_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)
    elif args.dataset_name == 'glue':
        train_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name, subset=args.task_name)
        train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
        
        test_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name,
                                                 subset=args.task_name, split='validation')
        test_loader = DataLoader(test_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)

    return train_dataset, train_loader, test_loader


def random_embedding_init(model, model_input, embedding_init):
    model.zero_grad()
    en = torch.norm(embedding_init, dim=-1, p=2).unsqueeze(-1)
    # embeddings = torch.randn_like(embedding_init)
    embeddings = embedding_init.clone()
    # embeddings = embeddings / torch.norm(embeddings, dim=-1, p=2).unsqueeze(-1)
    # embeddings = embeddings * en
    embeddings.requires_grad_()
    for i in range(20):
        model.zero_grad()
        embeddings.requires_grad_()
        model_input['inputs_embeds'] = embeddings
        outputs = model(**model_input).logits
        _, labels = torch.max(outputs, dim=-1)

        confidence, _ = torch.max(F.softmax(outputs), dim=-1)

        if torch.min(confidence) > 0.85:
            break

        losses = F.cross_entropy(outputs, labels.squeeze(-1))
        losses.backward()
        embeddings_grad = embeddings.grad.clone().detach()

        embeddings = (embeddings - 0.05 * embeddings_grad).detach()
        embeddings = embeddings / torch.norm(embeddings, dim=-1, p=2).unsqueeze(-1)
        embeddings = embeddings * en

    return embeddings, labels


def uap_attack(attack_path, args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = AutoConfig.from_pretrained(attack_path)
    model = AutoModelForSequenceClassification.from_pretrained(attack_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(attack_path)
    model.to(device)
    model.eval()


    for param in model.parameters():
        param.requires_grad = False

    optimizer_grouped_parameters = [
        {
            "params": [],
            "weight_decay": 0,
        }
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.attack_lr,
        eps=args.adam_epsilon,
        correct_bias=args.bias_correction
    )

    # Use suggested learning rate scheduler
    num_training_steps = args.attack_steps
    warmup_steps = num_training_steps * args.warmup_ratio
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)

    avg_loss = utils.ExponentialMovingAverage()
    deltas0 = []
    deltas1 = []

    ###
    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)
    uap_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset='imdb',
                                              subset='none')

    for ep in range(args.perturb_num):
        train_dataset, train_loader, test_loader = load_data(tokenizer, args)

        ###
        uap_loader = DataLoader(uap_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)


        dev_accuracy, dev_loss = eval(model, test_loader, device)
        logger.info(f'\nnum of perturb: {ep}, '
                    f'Loss: {dev_loss}, '
                    f'Dev_Accuracy: {dev_accuracy}')

        # initialize delta
        # '--adv_init_mag', default=0.05
        embedding_init = torch.zeros([1, 1, 768], dtype=torch.float64)
        if args.adv_init_mag > 0:
            # '--adv_norm_type', default='l2'
            if args.adv_norm_type == 'l2':
                delta = torch.zeros_like(embedding_init).uniform_(-1, 1)
                delta_norm = torch.norm(delta.view(1, -1).float(), p=2, dim=1).detach()
                reweights = (args.adv_init_mag / delta_norm).view(-1, 1, 1)
                delta = (delta * reweights).detach()
            elif args.adv_norm_type == 'linf':
                delta = torch.zeros_like(embedding_init).uniform_(-args.adv_init_mag,
                                                                  args.adv_init_mag)
        else:
            delta = torch.zeros_like(embedding_init)
            reweights = 0
            norm_after = 0

        delta = delta.to(device)
        dev_accuracy, dev_loss, _ = evaluate(model, test_loader, device, delta, args.class_bound)
        logger.info(f'random_init, '
                    f'num of perturb: {ep}, '
                    f'Loss: {dev_loss}, '
                    f'Dev_Accuracy: {dev_accuracy}, '
                    f'\ndelta_norm: {reweights}, ')

        step = 0

        # pbar = tqdm(train_loader)
        pbar = tqdm(uap_loader)

        avg_loss = utils.ExponentialMovingAverage()
        for model_inputs, labels in pbar:
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)

            # for uap
            word_embedding_layer = model.get_input_embeddings()
            input_ids = model_inputs['input_ids']
            del model_inputs['input_ids']  # new modified
            attention_mask = model_inputs['attention_mask']
            embedding_init = word_embedding_layer(input_ids)

            uap_init, labels = random_embedding_init(model, model_inputs, embedding_init)
            uap_init.to(device)
            labels.to(device)

            input_mask = attention_mask.to(embedding_init)
            input_lengths = torch.sum(input_mask, 1)

            total_loss = 0.0

            # (0) forward
            delta.requires_grad_()
            mask_tensor = input_mask.unsqueeze(2)
            repeat_shape = mask_tensor.shape
            model_inputs['inputs_embeds'] = (delta.repeat(repeat_shape[0], repeat_shape[1], 1)
                                             * mask_tensor).to(torch.float32) + uap_init   # new modified
            # batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
            logits = model(**model_inputs).logits

            # (1) backward
            losses = F.cross_entropy(logits, labels.squeeze(-1))
            loss = torch.mean(losses)
            total_loss += loss.item()

            loss.backward()

            if step > args.attack_steps -1:
                break

            # (2) get gradient on delta
            delta_grad = delta.grad.clone().detach()

            # (3) update and clip
            lr = args.attack_lr * (1 - step / args.attack_steps)
            if args.adv_norm_type == "l2":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + lr * delta_grad / denorm).detach()
                if args.adv_max_norm > 0:
                    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                    reweights = (args.adv_max_norm / delta_norm).view(-1, 1, 1)
                    delta = (delta * reweights).detach()

            elif args.adv_norm_type == "linf":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + lr * delta_grad / denorm).detach()

            optimizer.step()
            scheduler.step()
            model.zero_grad()
            avg_loss.update(loss)

            if step % 100 == 0 or step == args.attack_steps-1:

                dev_accuracy, dev_loss, delta_type = evaluate(model, test_loader, device, delta, args.class_bound)
                logger.info(f'step: {step}, '
                            f'Loss: {avg_loss.get_metric(): 0.4f}, '
                            f'num of perturb: {ep}, '
                            f'\nLr: {optimizer.param_groups[0]["lr"]: .3e}, '
                            f'\nDev_Accuracy: {dev_accuracy}, '
                            f'\ndelta_norm: {delta_norm.detach()}, ')

            step += 1

        pp = delta.view(-1)
        if delta_type == 1:
            deltas1.append(pp.detach().cpu().numpy().tolist())
        elif delta_type == 0:
            deltas0.append(pp.detach().cpu().numpy().tolist())
    dump_path = os.path.join('results', args.dataset_name)
    if not os.path.exists(dump_path):
        os.mkdir(dump_path)
    with open(os.path.join(dump_path,'deltas0.json'), 'w') as File:
        json.dump(deltas0, File)
    with open(os.path.join(dump_path,'deltas1.json'), 'w') as File:
        json.dump(deltas1, File)




def main(args):
    set_seed(args.seed)
    model_dir = args.ckpt_dir

    log_file = os.path.join(model_dir, 'attack_INFO.log')
    logger.addHandler(logging.FileHandler(log_file))

    uap_attack(model_dir, args)

if __name__ == '__main__':

    level = logging.INFO
    logging.basicConfig(level=level)

    main(args)
