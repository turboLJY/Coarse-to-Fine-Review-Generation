import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence
from masked_cross_entropy import *
import itertools
import random
import math
import sys
import os
from tqdm import tqdm
from load import loadPrepareData
from load import SOS_ID, EOS_ID, PAD_ID
import pickle
import logging
logging.basicConfig(level=logging.INFO)
    
#############################################
# Prepare Training Data
#############################################
def indexesFromSentence(vocab, sentence):
    ids = []
    for word in sentence:
        word = word
        ids.append(vocab.topic2idx[word])
    return ids

def Padding(l, fillvalue=PAD_ID):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue)) 

def binaryMatrix(l, value=PAD_ID):
    m = []
    for i in range(len(l)):
        m.append([])
        for j in range(len(l[i])):
            if l[i][j] == PAD_ID:
                m[i].append(0)
            else:
                m[i].append(1) # mask = 1 if not padding
    return m

# return attribute index 
def inputVar(data, vocab, evaluation=False):
    
    attr = [[d[0], d[1], d[2]] for d in data]  # length: batch 
    attrVar = Variable(torch.LongTensor(attr), volatile=evaluation) # (batch_size, attribute_num)

    return attrVar 

# convert to index, add zero padding
# return output variable, mask, max length of the sentences in batch
def outputVar(l, vocab):
    topic_input = [indexesFromSentence(vocab, sentence[:-1]) for sentence in l]
    topic_output = [indexesFromSentence(vocab, sentence[1:]) for sentence in l]
    inpadList = Padding(topic_input)
    outpadList = Padding(topic_output)
    mask = binaryMatrix(inpadList)
    mask = Variable(torch.ByteTensor(mask))
    inpadVar = Variable(torch.LongTensor(inpadList))
    outpadVar = Variable(torch.LongTensor(outpadList))
    return inpadVar, outpadVar, mask

# pair_batch is a list of (input, output) with length batch_size
# sort list of (input, output) pairs by output length, reverse input
def batch2TrainData(vocab, pair_batch, evaluation=False):
    pair_batch.sort(key=lambda x: len(x[1]), reverse=True) # sort on topic number
    input_batch, output_batch = [], []
    for i in range(len(pair_batch)):
        input_batch.append(pair_batch[i][0])
        output_batch.append(pair_batch[i][1])
    attr_input = inputVar(input_batch, vocab, evaluation=evaluation)
    topic_input, topic_output, mask = outputVar(output_batch, vocab) # convert sentence to ids and padding
    return attr_input, topic_input, topic_output, mask

