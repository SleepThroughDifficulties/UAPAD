import argparse
import functools
import logging
import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
import sys
import pandas
from tqdm import tqdm
import detect_utils.ood_metric as ood_metric

sys.path.append("..")

import torch

from torch.utils.data import DataLoader
from detect_utils.detect import detect, evaluate
import detect_utils.utils as utils

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def parse_args():
    parser = argparse.ArgumentParser()
    # settings
    parser.add_argument('--model_name', type=str, default='/workspace/saved_models/finetune/imdb/none/finetune_bert-base-uncased_imdb_lr2e-05_epochs3/epoch2')
    
    parser.add_argument('--attack_name', type=str, default='pwws') # textfooler, pwws, bertattack, textbugger

    parser.add_argument('--model_type', type=str, default='bert')
    parser.add_argument("--dataset_name", default='imdb', type=str) # sst2, imdb, ag_news
    parser.add_argument("--scenario", default='easy', choices=['easy', 'hard'], type=str)

    parser.add_argument('--delta_weight', default=0.55, type=float, help='delta*delta_weight+embedding_init')
    parser.add_argument('--delta0_index', default=1, type=int)
    parser.add_argument('--delta1_index', default=1, type=int)

    # hyper-parameters
    parser.add_argument('--do_search', action='store_true')
    # --do_search is for seaching hyper-para or detecting using the best hyper-para
    parser.add_argument('--max_seq_length', type=int, default=256)
    parser.add_argument('--bsz', type=int, default=16)
    parser.add_argument('--eval_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('-f', '--force_overwrite', default=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--cuda_device', type=int, default=3)


    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.cuda_device}"

    if args.dataset_name == 'sst2' or args.dataset_name == 'imdb':
        args.num_labels = 2
    elif args.dataset_name == 'ag_news':
        args.num_labels = 4
    else:
        assert KeyError

    if args.dataset_name == 'sst2':
        args.dataset_name = 'glue'
        args.task_name = 'sst2'
    else:
        args.task_name = 'none'


    test_or_dev = 'test' if args.do_search else 'test'
    args.tmp_dataset_name = args.dataset_name if args.dataset_name != 'glue' else 'sst2'
    if 'roberta' in args.model_name:
        args.model_type = 'roberta'
    else:
        args.model_type = 'bert'

    args.adv_file_path = f'/workspace/dataset/1500/{args.model_type}_dataset/{args.tmp_dataset_name}/tsv-format/{args.attack_name}_{test_or_dev}.tsv'
    args.dev_file_path = f'/workspace/dataset/1500/{args.model_type}_dataset/{args.tmp_dataset_name}/tsv-format/dev.tsv'
    args.test_file_path = f'/workspace/dataset/1500/{args.model_type}_dataset/{args.tmp_dataset_name}/tsv-format/test.tsv'

    args.detect_result_path = f'/workspace/detectResult/{args.model_type}/{args.dataset_name}/{args.task_name}/{args.attack_name}/'
    os.makedirs(args.detect_result_path, exist_ok=True)

    import time
    tt = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    args.log_path = f'{tt}_detectlog.log'

    return args


def single_detect(
    args, device, config, tokenizer, model, 
    train_loader, test_loader, adv_loader, log_file
    ):

    # test feature 这里数据集没有合并, 只能分别进行处理
    print('Detect Clean Test Examples')
    clean_outputs = detect(args, model, tokenizer, train_loader, test_loader, device)

    # clean_embeddings0 = clean_outputs['embeddings0']
    # clean_embeddings1 = clean_outputs['embeddings1']
    # clean_label = clean_outputs['labels']


    # debug:
    if args.delta_weight == 1.0:
        dev_accuracy = evaluate(args, model, test_loader, device)
        log_file.write(f'\nDEBUG: dev_acc: {dev_accuracy}\n')

    # adv feature
    print('Detect Adversarial Examples')
    adv_outputs = detect(args, model, tokenizer, train_loader, adv_loader, device)

    # adv_embeddings0 = adv_outputs['embeddings0']
    # adv_embeddings1 = adv_outputs['embeddings1']
    # adv_label = adv_outputs['labels']
    #
    #
    # km = pd.DataFrame()
    # km['0'] = [torch.norm(i).cpu().detach().numpy() for i in clean_embeddings0]
    # km['1'] = [torch.norm(i).cpu().detach().numpy() for i in clean_embeddings1]
    # km['labels'] = [i.cpu().detach().numpy() for i in clean_label]
    # km.to_csv('clean.csv', index=None)
    #
    # ad = pd.DataFrame()
    # ad['0'] = [torch.norm(i).cpu().detach().numpy() for i in adv_embeddings0]
    # ad['1'] = [torch.norm(i).cpu().detach().numpy() for i in adv_embeddings1]
    # ad['labels'] = [i.cpu().detach().numpy() for i in adv_label]
    # ad.to_csv('adv.csv', index=None)

    orig_adv_samples_nums = len(adv_outputs['pred_labels'])

    
    test_predicts = torch.tensor(clean_outputs['pred_labels'])
    adv_predicts = torch.tensor(adv_outputs['pred_labels'])
        
    # save log
    tmp_all_acc = tmp_test_acc = tmp_adv_acc = 0

    test_correct_cnt = (test_predicts == torch.zeros(test_predicts.size())).sum().item()
    adv_correct_cnt = (adv_predicts == torch.ones(adv_predicts.size())).sum().item()



    tp = adv_correct_cnt
    fn = adv_predicts.size(0) - tp
    tn = test_correct_cnt
    fp = test_predicts.size(0) - test_correct_cnt
    try: 
        recall = float(tp / (tp + fn))
    except:
        recall = 0.0
    try:
        precision = float(tp / (tp + fp))
    except: 
        precision = 0.0
    try:
        FF1 = float(2*precision*recall/(precision+recall))
    except: 
        FF1 = 0.0
    
    # log_file.write(f'FF1 = {FF1}\n')
    # log_file.write(f'tp = {tp}\n')
    # log_file.write(f'fn = {fn}\n')
    # log_file.write(f'tn = {tn}\n')
    # log_file.write(f'fp = {fp}\n')

    tmp_test_acc = float(test_correct_cnt / test_predicts.size(0))
    tmp_adv_acc = float(adv_correct_cnt / adv_predicts.size(0))
    tmp_all_acc = float(
        (test_correct_cnt + adv_correct_cnt) / (test_predicts.size(0) + adv_predicts.size(0))
        )

    if True:
        log_file.write('\nINFO: -------PARTING LING-------PARTING LING-------PARTING LING-------\n')
        log_file.write('INFO: ----------------args----------------\n')
        for k in list(vars(args).keys()):
            log_file.write(f'INFO: {k}: {vars(args)[k]}\n')
        log_file.write('INFO: ----------------args----------------\n')
        log_file.write('INFO: import paras: \n')
        log_file.write('INFO: ================RESULTS BEGIN================\n')
        log_file.write(
            f'\ntest_acc: {tmp_test_acc*100}\nadv_acc: {tmp_adv_acc*100}\nAccuracy: {tmp_all_acc*100}\n'
        )
        log_file.write(
            f'orig adv samples nums: {orig_adv_samples_nums}\n'
        )
        log_file.write(
            f'succ attack adv samples nums: {adv_predicts.size(0)}\nclean samples nums: {test_predicts.size(0)}\n'
        )
        log_file.write(
            f'tp: {tp}\nfn: {fn}\ntn: {tn}\nfp: {fp}\n'
        )
        log_file.write(
            f'recall: {recall*100}\nprecision: {precision*100}\nAccuracy: {tmp_all_acc*100}\nf1: {FF1*100}\n'
        )
        log_file.write('INFO: ================RESULTS END================\n')


def main(args):
    set_seed(args.seed)

    # pre-trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config, tokenizer, model = load_pretrained_models(args.model_name, args.model_type, args)
    model.to(device)
    # datasets
    train_loader, test_loader, adv_loader = load_datasets(args, tokenizer)

    # save results including: log, figure, predicts and criteria
    tmp_save_path = args.detect_result_path
    os.makedirs(tmp_save_path, exist_ok=True)
    print(f'debugger: {os.path.join(tmp_save_path, args.log_path)}')
    f_out = open(os.path.join(tmp_save_path, args.log_path), 'w+')
    # 搜索hyper-parameters
    if args.do_search:
        hyper_weight_list = [0.1 * i for i in range(1, 21)]
        print(hyper_weight_list)
        for ww in hyper_weight_list:
            args.delta_weight = ww
            single_detect(
                args, device, config, tokenizer, model, 
                train_loader, test_loader, adv_loader, f_out
            )
    else:
        single_detect(
            args, device, config, tokenizer, model, 
            train_loader, test_loader, adv_loader, f_out
        )



def load_datasets(args, tokenizer):
    # dataloader
    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)
    # for training and dev
    train_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name, subset=args.task_name)
    train_size = round(int(len(train_dataset)))

    # train and dev dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)

    test_dataset = utils.local_dataset(args, tokenizer, dataset_path=args.test_file_path, dataset_name=args.tmp_dataset_name)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)

    adv_dataset = utils.local_dataset(args, tokenizer, dataset_path=args.adv_file_path, dataset_name=args.tmp_dataset_name)
    adv_loader = DataLoader(adv_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)

    return train_loader, test_loader, adv_loader


def load_pretrained_models(model_name, model_type, args):
    if model_type == 'bert' or model_type == 'roberta':
        # print(f'debugger: {model_name}')        
        config = AutoConfig.from_pretrained(model_name, num_labels=args.num_labels)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    return config, tokenizer, model


def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':


    args = parse_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    main(args)
