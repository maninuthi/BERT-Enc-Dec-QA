from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
from gensim.models import word2vec
import gensim

from collections import Counter
from nltk import word_tokenize
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd

import os
import re
import random
import nltk

# use_cuda = config.use_gpu and torch.cuda.is_available()

import tensorflow as tf

import time
start = time.time()

config = tf.ConfigProto(intra_op_parallelism_threads = 1, inter_op_parallelism_threads = 1)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config=config)
from keras import backend as K
K.set_session(sess)


from transformers import BertModel, BertTokenizer, BertForQuestionAnswering

#Creating instance of BertModel
bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')

#Creating intance of tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


hidden_size = 768
hidden_dim = 50
layer_dim = 1
batch_size = 1
max_tot_len = 500
max_ques_len = 90
stride_len = 90


def formulate(data, s, e):
    ans= ''
    i=int(s)
    try:
        while(i<=int(e)):
            ans = ans+' '+data[i]
            i+=1
    except:
        return 0

    return ans
    

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(pred , grd):

    if (pred.isalpha() or grd.isalpha()):

        prediction_tokens = pred.split()
        ground_truth_tokens = grd.split()

    else:
        prediction_tokens = word_tokenize(pred)
        ground_truth_tokens = word_tokenize(grd)
    
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    #print(precision)
    return f1       


def prepare_data(para_tokens,ques_tokens,start,end):
    all_tokens_len = 0
    sub_token_ind = []
    subtokens = []
    
    if(len(ques_tokens)> max_ques_len):
        ques_tokens = ques_tokens[:max_ques_len]
    
    max_para_len = max_tot_len - len(ques_tokens) - 2 
    
    fixture = len(tokenizer.tokenize(para_tokens[end])) -1 
    
    chunk_start = []
    inpt= []
    sid = []
    start_ind = []
    end_ind = []
    some = 0
    sub_token_ind.append(0)

    for para_token in para_tokens:
        temp_t = tokenizer.tokenize(para_token)
        subtokens+= temp_t
        sub_token_ind.append(len(subtokens))
        
    while(some <= len(subtokens)):
        chunk_start.append(some)
        if(sub_token_ind[start] >= some and sub_token_ind[end]+fixture < min(some+max_para_len,len(subtokens)) ):
            start_ind.append(sub_token_ind[start]-some + len(ques_tokens) + 2)
            end_ind.append(sub_token_ind[end]-some + fixture + len(ques_tokens) + 2)
            sid.append([0]*(len(ques_tokens)+2) + [1]*(len(subtokens[some:min(some+max_para_len,len(subtokens))])+1))
            inpt.append(['[CLS]']+ques_tokens+['[SEP]']+subtokens[some:min(some+max_para_len,len(subtokens))]+['[SEP]'])
            
        if(some+max_para_len > len(subtokens)):
            return inpt,sid,start_ind,end_ind
        some+= stride_len
        
def get_data(file_path1,file_path2,file_path3):
    with open(file_path1) as f1, open(file_path2) as f2,open(file_path3) as f3:
        para = f1.readlines()
        ques = f2.readlines()
        span = f3.readlines()
        
    inpt = []
    sid = []
    msk = []
    start = []
    end = []
    for i in range(1000):
        para_tokens = word_tokenize(para[i])
        ques_tokens = tokenizer.tokenize(ques[i])
        temp = (span[i].strip()).split('\t')
        
        tempr,ids,s,e =prepare_data(para_tokens,ques_tokens,int(temp[0]),int(temp[1]))
        for j in range(len(tempr)):
            input_ids = tokenizer.convert_tokens_to_ids(tempr[j])
            
            inpt.append(input_ids)
            sid.append(ids[j])
            start.append(s[j])
            end.append(e[j])   
    
    return inpt,sid,start,end
        

inputs,sid,start,end = get_data("para","ques","asnr")



class Highway(nn.Module):

    def __init__(self, input_size):
        super(Highway, self).__init__()
        self.proj = nn.Linear(input_size, input_size)
        self.transform = nn.Linear(input_size, input_size)
        self.transform.bias.data.fill_(-2.0)

    def forward(self, input):
        proj_result = nn.functional.relu(self.proj(input))
        proj_gate = nn.functional.sigmoid(self.transform(input))
        gated = (proj_gate * proj_result) + ((1 - proj_gate) * input)
        return gated

class QA(nn.Module):
    def __init__(self, hidden_size , hidden_dim, layer_dim, batch_size):
        super(QA, self).__init__()
        
        # self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.hidden_dim = hidden_dim
        self.highway = Highway(hidden_dim*2)
        self.hidden_size = hidden_size
        self.layer_dim = layer_dim
        self.batch_size = batch_size
        self.gen_span = nn.Linear(hidden_dim*2,2)
        self.lstm = nn.LSTM(hidden_size, hidden_dim, layer_dim, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_dim*2, hidden_dim, layer_dim, batch_first=True, bidirectional=True)

#         self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x,y):
        
#         inp_len = len(x)
#         x = x.view(1,inp_len,self.hidden_size)
 
        seq ,_ = bert_model(x, token_type_ids = y)
        h0 = torch.zeros(self.layer_dim*2, self.batch_size, self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layer_dim*2, self.batch_size, self.hidden_dim).requires_grad_()

        out, (hn, cn) = self.lstm(seq, (h0.detach(), c0.detach()))
        # print(out.shape)
        out = self.highway(out[0])
        # print(out.shape)
        out = out.unsqueeze(0)
        # print(out.shape)
        h1 = torch.zeros(self.layer_dim*2, self.batch_size, self.hidden_dim).requires_grad_()
        c1 = torch.zeros(self.layer_dim*2, self.batch_size, self.hidden_dim).requires_grad_()

        out, (hn, cn) = self.lstm2(out, (h1.detach(), c1.detach()))
        # print(out.shape)
        temp = self.gen_span(out[0])
        
        start_logits, end_logits = temp.split(1, dim=-1)
#         start_logits = start_logits.squeeze(-1)
#         end_logits = end_logits.squeeze(-1)
        
        return start_logits, end_logits
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_dim, device=device)

model = QA(hidden_size,hidden_dim,layer_dim,batch_size)
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train(model,src_data,sid,optimizer,start,end):
    total_loss = 0
    for i in range(len(src_data)):
        optimizer.zero_grad()
        x = torch.tensor([src_data[i]])
        y = torch.tensor([sid[i]])
        
        start_logits, end_logits = model(x,y)
        
        ignored_index = len(start_logits)
#         start_positions.clamp_(0, ignored_index)
#         end_positions.clamp_(0, ignored_index)
        #torch.Size([1]) torch.Size([1, 16]) tensor([0]) tensor([[-0.0493, -0.0218,  0.0876,  0.3616,  0.0129,  0.6303,  0.3034, -0.3084,0.1309,  0.6360,  0.2071, -0.0658,  0.1549,  0.5422, -0.3890, -0.3889]], grad_fn=<SqueezeBackward1>)
        start_logits = start_logits.view(1,-1)
        start_position = torch.tensor([start[i]])
        end_logits = end_logits.view(1,-1)
        end_position = torch.tensor([end[i]])
        
        loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_position)
        end_loss = loss_fct(end_logits, end_position)
        loss = (start_loss + end_loss) / 2
        total_loss+=loss
        loss.backward()
        optimizer.step()
    print(total_loss)

def test(model,src_data,sid,start,end):
    fh = open('gen_text','a')
    for i in range(len(src_data)):
        x = torch.tensor([src_data[i]])
        y = torch.tensor([sid[i]])
        start_logits, end_logits = model(x,y)
        _,sm = torch.max(start_logits,0)
        _,em = torch.max(end_logits,0)
        try:
            fh.write(str(sm)+'\t'+str(em)+'\t\t'+str(start[i])+'\t'+str(end[i])+'\t'+str(tokenizer.decode(src_data[i][int(sm):int(em)+1]))+'\t'+str(tokenizer.decode(src_data[i][int(start[i]):int(end[i])+1])))
        except:
            fh.write(str(sm)+'\t'+str(em)+'\t\t'+str(start[i])+'\t'+str(end[i])+'\t'+str(tokenizer.decode(src_data[i][int(start[i]):int(end[i])+1])))
        fh.write('\n')

for epoch in range(15):
    train(model,inputs,sid,optimizer,start,end)
test(model,inputs,sid,start,end)
