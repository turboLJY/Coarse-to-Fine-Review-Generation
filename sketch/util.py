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
import numpy as np
logging.basicConfig(level=logging.INFO)
    
#############################################
# Prepare Training Data
#############################################
def indexesFromTopic(vocab, sentence):
    ids = []
    for word in sentence:
        word = word
        ids.append(vocab.topic2idx[word])
    return ids

def indexesFromSketch(vocab, sentence):
    ids = []
    for word in sentence:
        word = word
        ids.append(vocab.sketch2idx[word])
    return ids

# batch_first: true -> false, i.e. shape: seq_len * batch
def Padding(l, fillvalue=PAD_ID):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue)) 

def binaryMatrix(l, value=PAD_ID):
    m = []
    for i in range(len(l)):  # max_topic
        m.append([])
        for j in range(len(l[i])):
            
            if l[i][j] == PAD_ID:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# return attribute index and input pack_padded_sequence
def inputVar(data, vocab, evaluation=False):
    
    attr = [[d[0], d[1], d[2]] for d in data]  # length: batch 
    attrVar = Variable(torch.LongTensor(attr), volatile=evaluation) # (batch_size, attribute_num), in our case it is 2

    return attrVar 

# convert to index, add zero padding
# return output variable, mask, max length of the sentences in batch
def SketchVar(pb, tb, vocab):

    topic = [indexesFromTopic(vocab, b) for b in tb]
    # manually padding
    sketch_input = []
    sketch_output = []
    topic_input = []
    for i in range(len(pb)):
        inp = []
        top = []
        out = []
        assert len(topic[i]) == len(pb[i])
        for j in range(len(pb[i])):
            sketch = pb[i][j]
            top += [topic[i][j]] * (len(sketch) - 1)
            inp += indexesFromSketch(vocab, sketch[:-1])
            out += indexesFromSketch(vocab, sketch[1:])
        assert len(top) == len(inp)
        topic_input.append(top)
        sketch_input.append(inp)
        sketch_output.append(out)

    topicList = Padding(topic_input)
    inpadList = Padding(sketch_input)
    outpadList = Padding(sketch_output)
    mask = binaryMatrix(outpadList)

    topicVar = Variable(torch.LongTensor(topicList))
    inpadVar = Variable(torch.LongTensor(inpadList))
    outpadVar = Variable(torch.LongTensor(outpadList))
    mask = Variable(torch.ByteTensor(mask))
    return topicVar, inpadVar, outpadVar, mask

# pair_batch is a list of (input, output) with length batch_size
# sort list of (input, output) pairs by output length, reverse input
# return input, lengths for pack_padded_sequence, output_variable, mask
def batch2TrainData(vocab, pair_batch, evaluation=False):
    pair_batch.sort(key=lambda x: len(x[1]), reverse=True) # sort on topic length
    input_batch, topic_batch, sketch_batch = [], [], []
    for i in range(len(pair_batch)):
        input_batch.append(pair_batch[i][0])
        topic_batch.append(pair_batch[i][1])
        sketch_batch.append(pair_batch[i][2])
    attr_input = inputVar(input_batch, vocab, evaluation=evaluation)
    topic_input, sketch_input, sketch_output, mask = sketchVar(sketch_batch, topic_batch, vocab)
    return attr_input, topic_input, sketch_input, sketch_output, mask 

