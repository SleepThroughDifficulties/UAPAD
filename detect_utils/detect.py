import sys
import os
import json

sys.path.append("..")
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def detect(args, model, tokenizer, train_loader, target_loader, device):

    # adv + delta feature
    with open(os.path.join('results', args.dataset_name, 'deltas0.json'), 'r') as f1:
        delta0s = json.load(f1)
    with open(os.path.join('results', args.dataset_name, 'deltas1.json'), 'r') as File2:
        delta1s = json.load(File2)


    true_labels, check_labels, check_logits, pred_labels, pred_logits = None, None, None, None, None

    pbar = tqdm(target_loader)
    # pbar = tqdm(train_loader)
    embeddings0 = []
    embeddings1 = []
    label_list = []

    for model_inputs, labels in pbar:
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

        labels = labels.to(device)
        model.zero_grad()
        word_embedding_layer = model.get_input_embeddings()
        input_ids = model_inputs['input_ids']
        del model_inputs['input_ids']  # new modified
        attention_mask = model_inputs['attention_mask']
        embedding_init = word_embedding_layer(input_ids)

        input_mask = attention_mask.to(embedding_init)
        input_lengths = torch.sum(input_mask, 1)
        mask_tensor = input_mask.unsqueeze(2)
        repeat_shape = mask_tensor.shape    

        delta0 = torch.tensor(delta0s[args.delta0_index]).to(device)
        delta1 = torch.tensor(delta1s[args.delta1_index]).to(device)


        model_inputs['inputs_embeds'] = embedding_init
        logits_check = model(**model_inputs).logits
        _, preds_check = logits_check.max(dim=-1)

        batch_delta = torch.zeros(embedding_init.size()).to(device)

        for i in range(preds_check.size()[0]):
                if preds_check[i] == 1:
                    batch_delta[i,:,:] = -delta1.repeat(repeat_shape[1], 1)
                    # batch_delta[i, :, :] = delta0.repeat(repeat_shape[1], 1)
                elif preds_check[i] == 0:
                    batch_delta[i,:,:] = -delta0.repeat(repeat_shape[1], 1)
                    # batch_delta[i, :, :] = delta1.repeat(repeat_shape[1], 1)
                else:
                    assert ValueError

        #TODO test
        pp = embedding_init * mask_tensor
        for i in range(pp.shape[0]):
            embd0 = torch.matmul(pp[i,:,:].squeeze(0), delta0) / torch.norm(delta0)
            embd1 = torch.matmul(pp[i,:,:].squeeze(0), delta1) / torch.norm(delta1)
            embeddings0.append(embd0)
            embeddings1.append(embd1)
            label_list.append(labels[i])


        model_inputs['inputs_embeds'] = embedding_init + (args.delta_weight*batch_delta*mask_tensor).to(torch.float32)

        logits = model(**model_inputs).logits
        _, preds = logits.max(dim=-1)


        if pred_logits is None:
            predict_logits = torch.softmax(logits, dim=1).detach().cpu()
            check_logits = torch.softmax(logits_check, dim=1).detach().cpu()
        else:
            predict_logits = torch.cat((predict_logits, torch.softmax(logits, dim=1).detach().cpu()), dim=0)
            check_logits = torch.cat((check_logits, torch.softmax(logits_check, dim=1).detach().cpu()), dim=0)

        if pred_labels is None:
            pred_labels = preds.detach().cpu().numpy()
            check_labels = preds_check.detach().cpu().numpy()
            true_labels = labels.detach().cpu().numpy()
        else:
            pred_labels = np.append(pred_labels, preds.detach().cpu().numpy(), axis=0)
            check_labels = np.append(check_labels, preds_check.detach().cpu().numpy(), axis=0)
            true_labels = np.append(true_labels, labels.detach().cpu().numpy(), axis=0)

    
    adv_pred_labels = np.array([
        0 if pred_labels[i] == check_labels[i] else 1 for i in range(len(true_labels))
        ])

    return {
        'true_labels': true_labels,
        'check_labels': check_labels,
        'check_logits': check_logits,
        'pred_labels': adv_pred_labels,
        'pred_logits': pred_logits,
        'embeddings0': embeddings0,
        'embeddings1': embeddings1,
        'labels': label_list
    }




def evaluate(args, model, data_loader, device):

    with open(os.path.join('results', args.dataset_name, 'deltas0.json'), 'r') as f1:
        delta0s = json.load(f1)
    delta = torch.tensor(delta0s[args.delta0_index]).to(device)

    model.eval()
    correct = 0
    total = 0
    pred_label = 0
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
            pred_label += (preds == 1).sum().item()
            correct += (preds == labels.squeeze(-1)).sum().item()
            total += labels.size(0)
        accuracy = correct / (total + 1e-13)

    return accuracy