import pandas as pd
import numpy as np
from collections import Counter
from transformers import BertPreTrainedModel,RobertaModel,RobertaPreTrainedModel,   AdamW,  get_linear_schedule_with_warmup
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch
import torch.nn as nn
import json
import os
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from datasets import load_dataset, load_metric
from transformers import DataCollatorForTokenClassification
from transformers import BertTokenizer,RobertaTokenizer, AutoTokenizer,AutoConfig
# 设置随机种子，便于复现结果以及避免随机使得模型之间更加可比
import random
SEED = 2020
import argparse
from torchcrf import CRF



class RobertaSoftmaxForNer(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RobertaSoftmaxForNer, self).__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_type = config.loss_type
        self.config = config
        self.post_init()

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            valid_mask=None,
            labels=None
    ):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.loss_type == 'ce':
                loss_fct = CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)

class RobertaLSTMCRF(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RobertaLSTMCRF, self).__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.lstm = nn.LSTM(input_size=config.hidden_size,
                                hidden_size=config.hidden_size//2,
                                num_layers=1,
                                dropout=config.hidden_dropout_prob,
                                bidirectional=True,
                                batch_first=True)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.config = config
        self.crf = CRF(num_tags=config.num_labels,batch_first=True)
        self.post_init()
        
    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            valid_mask=None,
            labels=None
    ):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        packed_embed_out = nn.utils.rnn.pack_padded_sequence(sequence_output, torch.sum(attention_mask,dim=-1).cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, (h_n, c_n) = self.lstm(packed_embed_out)
        sequence_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        logits = self.classifier(sequence_output)
        loss = None
        mask = attention_mask > 0
        if labels is not None:
            labels = labels * (labels != -100).int()
            log_likelihood = self.crf(logits, labels, mask=mask)
            loss = 0 - log_likelihood
        tags = self.crf.decode(logits)
        tags = torch.Tensor(tags).int()
        output = (tags,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output      
    
class RobertaCRF(RobertaPreTrainedModel):
    def __init__(self,config):
        super(RobertaCRF, self).__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.config = config
        self.crf = CRF(num_tags=config.num_labels,batch_first=True)
        self.post_init()
    
    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            valid_mask=None,
            labels=None
    ):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss = None
        mask = attention_mask > 0
        if labels is not None:
            labels = labels * (labels != -100).int()
            log_likelihood = self.crf(logits, labels, mask=mask)
            loss = 0 - log_likelihood
        tags = self.crf.decode(logits)
        tags = torch.Tensor(tags).int()
        output = (tags,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output    

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights = None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
            )

    def forward(self, all_hidden_states):
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average
    
class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''

    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight, ignore_index=self.ignore_index)
        return loss
    
def tokenize_and_align_labels(examples):
#     print(examples["tokens"])
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True,max_length=128)

    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(b_to_i_label[label2id[label[word_idx]]])
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def get_labels(predictions, references):
    # Transform predictions and references tensos to numpy arrays
    if device == "cpu":
        y_pred = predictions.detach().clone().numpy()
        y_true = references.detach().clone().numpy()
    else:
        y_pred = predictions.detach().cpu().clone().numpy()
        y_true = references.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
        for pred, gold_label in zip(y_pred, y_true)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
        for pred, gold_label in zip(y_pred, y_true)
    ]
    return true_predictions, true_labels

def compute_metrics(return_entity_level_metrics=False):
    results = metric.compute(zero_division=0)
    if return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
def eval_model(model, dataset):
    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    for i, batch in enumerate(dataset):
        with torch.no_grad():
            # Forward pass
            batch = {k:v.to(device) for k,v in batch.items()}
            outs = model(**batch)
            tmp_eval_loss, logits = outs[:2]
            predictions = logits.argmax(dim=-1) if not use_crf else logits
            labels = batch["labels"]
        preds, refs = get_labels(predictions, labels)
        metric.add_batch(
                    predictions=preds,
                    references=refs,
                )  # predictions and preferences are expected to be a nested list of labels, not label_ids

    eval_metric = compute_metrics()
    print(eval_metric)
    return eval_metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/opt/ml/input/data/pretrained_model/0_Transformer", help="")
    parser.add_argument("--label_list", default="/opt/ml/input/data/training/labels_list.json", help="")
    parser.add_argument("--train_path", default="/opt/ml/input/data/training/train.jsonl", help="")
    parser.add_argument("--dev_path", default="/opt/ml/input/data/training/dev.jsonl", help="")
    parser.add_argument("--epochs", type=int, default=10, help="")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="")
    parser.add_argument("--warmup_ratio", type=float, default=0.06, help="")
    parser.add_argument("--batch_size", type=int, default=32, help="")
    parser.add_argument("--max_length", type=int, default=128, help="")
    parser.add_argument("--model_save", default='/opt/ml/model', help="")
    args=parser.parse_args()
    
    model_path=args.model_path #"roberta-base"#"/home/ec2-user/SageMaker/Shulex/场景抽取/ConSERT/output/unsup-consert-base/0_Transformer"
    label_list = json.load(open(args.label_list))
    label_list+=['B-场景-Moment','I-场景-Moment']
    label2id = {lb:i for i,lb in enumerate(label_list)}
    id2label = {v:k for k,v in label2id.items()}
    num_labels = len(label_list)
    tokenizer = AutoTokenizer.from_pretrained(model_path,add_prefix_space=True)
    
    max_length = args.max_length #256
    batch_size = args.batch_size
#     print(type(batch_size))
    use_crf = True
    device = 'cuda'
    learning_rate = args.learning_rate
    gradient_accumulation_steps = 1
    epochs = args.epochs
    warmup_ratio = args.warmup_ratio
    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in label_list:
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)
    dataset = load_dataset('json', data_files={'train':args.train_path, 'validation':args.dev_path})
    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True,remove_columns=dataset["train"].column_names)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    train_dataloader = DataLoader(
        tokenized_datasets['train'], shuffle=True, collate_fn=data_collator, batch_size=batch_size
    )
    eval_dataloader = DataLoader(tokenized_datasets['validation'], collate_fn=data_collator, batch_size=batch_size)
    config = AutoConfig.from_pretrained(model_path,num_labels=num_labels)#Shulex/场景抽取/ConSERT/output/unsup-consert-base/0_Transformer
    model = RobertaCRF.from_pretrained(model_path,config=config).to(device)
    # setting custom optimization parameters. You may implement a scheduler here as well.
    param_optimizer = list(model.named_parameters())
    crf_params = [(n,p) for n,p in param_optimizer if 'roberta.' not in n]
    param_optimizer = [(n,p) for n,p in param_optimizer if 'crf.' not in n]
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0},
        {'params': [p for n,p in crf_params if 'crf.' in n],'lr':learning_rate*100}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,lr=learning_rate,correct_bias=True)
    
    t_total = len(train_dataloader) // gradient_accumulation_steps * epochs
    warmup_steps =  int(t_total * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=t_total,
                )
    metric = load_metric("seqeval")
    pre_best = -1
    save_best = True
    epoch_steps = len(train_dataloader)

    # trange is a tqdm wrapper around the normal python range
    for _ in range(epochs):
    # Training
    # Set our model to training mode (as opposed to evaluation mode)
        model.train()

    # Tracking variables
        tr_loss = 0 #running loss
        nb_tr_examples, nb_tr_steps = 0, 0

    # Train the data for one epoch
        for step, batch in enumerate(tqdm(train_dataloader)):
        # Add batch to GPU
            batch = {k:v.to(device) for k,v in batch.items()}
        
            outputs = model(**batch)
            loss = outputs[0]
            loss.backward()
        
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        # Update tracking variables
            tr_loss += loss.item()
            nb_tr_steps += 1

            if step == (epoch_steps // 2):
                eval_metric = eval_model(model,eval_dataloader)
                if save_best and eval_metric['f1'] > pre_best:
                    print('save better model')
                    pre_best = eval_metric['f1']
                    torch.save(model.state_dict(),os.path.join(args.model_save, 'model.bin'))
                model.train()

        print("Train loss: {}".format(tr_loss/nb_tr_steps))
        eval_metric = eval_model(model,eval_dataloader)
        if save_best and eval_metric['f1'] > pre_best:
            print('save better model')
            pre_best = eval_metric['f1']
            torch.save(model.state_dict(),os.path.join(args.model_save, 'model.bin'))
    model.load_state_dict(torch.load(os.path.join(args.model_save, 'model.bin')))
    print(eval_model(model,eval_dataloader))