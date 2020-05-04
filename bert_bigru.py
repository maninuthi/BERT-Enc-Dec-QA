from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
from gensim.models import word2vec
import gensim


from nltk import word_tokenize
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd
from collections import Counter
import os
import re
import random
import nltk

# use_cuda = config.use_gpu and torch.cuda.is_available()

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

from transformers import BertModel, BertTokenizer, BertForQuestionAnswering

#Creating instance of BertModel
bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')
bert_model = bert_model.to(device)
#Creating intance of tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


hidden_size = 768
hidden_dim = 50
layer_dim = 1
batch_size = 1
max_tot_len = 500
max_ques_len = 90
stride_len = 90

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
        

inputs,sid,start,end = get_data("para_tel","ques_tel","span_tel")

def normalize(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

#     def remove_punc(text):
#         exclude = set(string.punctuation)
#         return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(lower(s)))


def f1_score(pred , grd):

    if (pred.isalpha() or grd.isalpha()):

        prediction_tokens = normalize(pred).split()
        ground_truth_tokens = normalize(grd).split()

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

def exact_match_score(prediction, ground_truth):
    return (normalize(prediction) == normalize(ground_truth))



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
        self.gru = nn.GRU(hidden_size,hidden_dim, batch_first=True,bidirectional = True)
        self.gru2 = nn.GRU(hidden_dim*2,hidden_dim, batch_first=True,bidirectional = True)

#         self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x,y):
        
#         inp_len = len(x)
#         x = x.view(1,inp_len,self.hidden_size)
 
        seq ,_ = bert_model(x, token_type_ids = y)
        h0 = torch.zeros(self.layer_dim*2, self.batch_size, self.hidden_dim).requires_grad_().to(device)
        out, hn = self.gru(seq, h0.detach())
        out = out.to(device)
        h1 = torch.zeros(self.layer_dim*2, self.batch_size, self.hidden_dim).requires_grad_().to(device)
        out, hn = self.gru2(out, h1.detach())
        # print(out.shape)
        out = out.to(device)
        temp = self.gen_span(out[0]).to(device)
        
        start_logits, end_logits = temp.split(1, dim=-1)

#         start_logits = start_logits.squeeze(-1)
#         end_logits = end_logits.squeeze(-1)
        
        return start_logits, end_logits
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_dim, device=device)

model = QA(hidden_size,hidden_dim,layer_dim,batch_size).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train(model,src_data,sid,optimizer,start,end):
    total_loss = 0
    for i in range(len(src_data)):
        optimizer.zero_grad()
        x = torch.tensor([src_data[i]])
        y = torch.tensor([sid[i]])

        x = x.to(device)
        y = y.to(device)
        
        start_logits, end_logits = model(x,y)
        
        ignored_index = len(start_logits)
#         start_positions.clamp_(0, ignored_index)
#         end_positions.clamp_(0, ignored_index)
        #torch.Size([1]) torch.Size([1, 16]) tensor([0]) tensor([[-0.0493, -0.0218,  0.0876,  0.3616,  0.0129,  0.6303,  0.3034, -0.3084,0.1309,  0.6360,  0.2071, -0.0658,  0.1549,  0.5422, -0.3890, -0.3889]], grad_fn=<SqueezeBackward1>)
        start_logits = start_logits.view(1,-1)
        start_logits = start_logits.to(device)
        start_position = torch.tensor([start[i]])
        start_position = start_position.to(device)
        end_logits = end_logits.view(1,-1)
        end_logits = end_logits.to(device)
        end_position = torch.tensor([end[i]])
        end_position = end_position.to(device)

        loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_position)
        end_loss = loss_fct(end_logits, end_position)
        loss = (start_loss + end_loss) / 2
        total_loss+=loss
        loss.backward()
        optimizer.step()
    print(total_loss)

def test(model,src_data,sid,start,end):
    f1 = 0
    em = 0
    fh = open('15gru','a')
    for i in range(len(src_data)):
        x = torch.tensor([src_data[i]])
        y = torch.tensor([sid[i]])

        x = x.to(device)
        y = y.to(device)

        start_logits, end_logits = model(x,y)
        _,sm = torch.max(start_logits,0)
        _,em = torch.max(end_logits,0)
        if(sm>em):
            temp = sm
            em = sm
            sm = temp
        pred = str(tokenizer.decode(src_data[i][int(sm):int(em)+1]))
        actual = str(tokenizer.decode(src_data[i][int(start[i]):int(end[i])+1]))
        f1+= f1_score(pred,actual)
        if(exact_match_score(pred,actual)):
            em+=1
        fh.write(str(pred+'\t'+actual))
        fh.write('\n')
    print("F1_Score",f1/len(sid))
    print("EM_Score",em/len(sid))

for epoch in range(30):
    train(model,inputs[:900],sid[:900],optimizer,start[:900],end[:900])
test(model,inputs[900:],sid[900:],start[900:],end[900:])

torch.save(model,'grumodel')

# m = torch.load('grumodel')
# m = m.to(device)
# def test_(model,src_data,sid):
#     fh = open('15combi','a')
#     for i in range(len(src_data)):
#         x = torch.tensor([src_data[i]])
#         y = torch.tensor([sid[i]])

#         x = x.to(device)
#         y = y.to(device)

#         start_logits, end_logits = model(x,y)
#         _,sm = torch.max(start_logits,0)
#         _,em = torch.max(end_logits,0)
#         if(sm>em):
#             temp = sm
#             em = sm
#             sm = temp
#         print(str(sm)+'\t'+str(em)+'\t\t'+str(tokenizer.decode(src_data[i][int(sm):int(em)+1])))


# test_(m,inputs[910:911],sid[910:911])