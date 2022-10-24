from transformers import BertPreTrainedModel,RobertaModel,RobertaPreTrainedModel,   AdamW,  get_linear_schedule_with_warmup
from main import *
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import os
import csv
import pickle
import itertools
import time
import numpy as np
import logging
import json
import torch

model_name = 'roberta-base'
num_labels=9
model_path=''

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def model_fn(model_dir):
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name,add_prefix_space=True)
    config = AutoConfig.from_pretrained(model_name,num_labels=num_labels)
    model = RobertaCRF.from_pretrained(model_name,config=config).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'code', 'model.bin')))
    model.eval()
    return model, tokenizer

# def input_fn(json_request_data, content_type='application/json'):  
#     input_data = json.loads(json_request_data)
#     text_to_summarize = input_data['text']
#     return text_to_summarize
def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """
    
    if request_content_type == "application/json":
        request = json.loads(request_body)
    else:
        request = request_body

    return request


def predict_fn(input_data, model):
    device = get_device()
    model, tokenizer=model
    inp=tokenizer([input_data['text']], truncation=True, is_split_into_words=True,max_length=512,return_tensors="pt").to(device)
#     sentence = Sentence(input_data)
    pred = model(**inp)

    res={'res': list(pred[0].cpu().numpy()[0])}
        
    return res


def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device

# def output_fn(embedding, accept='application/json'):
#     return json.dumps({'features':embedding}, cls=NpEncoder), accept
def output_fn(prediction, response_content_type):
    """
    Serialize and prepare the prediction output
    """
    
    if response_content_type == "application/json":
        response = str(prediction)
    else:
        response = str(prediction)

    return response

if __name__ == '__main__':

    input_data ={'text':'as as'}
    
    model = model_fn('/home/ec2-user/SageMaker/Shulex/场景抽取/ConSERT/sagemaker/code')
    result = predict_fn(input_data, model)
#     print(json.dumps({'features': result}, cls=NpEncoder))
    print(result)