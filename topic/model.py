# coding=utf-8
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
        
class AttributeEncoder(nn.Module):
    def __init__(self, attr_size, attr_num, hidden_size, attr_embeddings, n_layers=1):
        super(AttributeEncoder, self).__init__()
        self.attr_size = attr_size
        self.attr_num = attr_num  
        self.hidden_size = hidden_size
        self.n_layers = n_layers 
        self.user_embedding = attr_embeddings[0]
        self.item_embedding = attr_embeddings[1]
        self.over_embedding = attr_embeddings[2]
        # hidden matrix is L*n, where L is number of layers and n is hidden size of each unit
        self.attr = nn.Linear(self.attr_size * self.attr_num, self.n_layers * self.hidden_size)  # used to initialize rnn 
        self.tanh = nn.Tanh()

    def forward(self, input):  # input size: (batch, attr_num)
        embeddeds = [] # (attr_num, batch, attr_size) = (A,B,K)
        embeddeds.append(self.user_embedding(input[:, 0]))  # (batch) --> (batch, attr_size)
        embeddeds.append(self.item_embedding(input[:, 1]))  # (batch) --> (bacth, attr_size)
        embeddeds.append(self.over_embedding(input[:, 2]))  # (batch) --> (bacth, attr_size)

        embedded = torch.cat(embeddeds, 1) # (batch, attr_num * attr_size)
        hidden = self.tanh(self.attr(embedded)).contiguous() # (batch, n_layer*hidden_size)
       
        # used as initialized hidden state
        hidden = hidden.view(-1, self.n_layers, self.hidden_size).transpose(0, 1).contiguous() # (layer, B, hidden)

        # used as the representation of user, item and overall 
        output = torch.stack(embeddeds) # [(B,K), (B,K), (B, K)] list to tensor (A,B,K), default in dim=0
        return output, hidden

class AttributeAttn(nn.Module):
    def __init__(self, hidden_size, attr_size):
        super(AttributeAttn, self).__init__()
        self.hidden_size = hidden_size
        self.attr_size = attr_size
        self.attn = nn.Linear(self.hidden_size + attr_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        '''
        :param hidden: 
            previous hidden state of the decoder, in shape (N,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (A,B,K) = (# of attributes, batch size, attribute dimension)
        :return
            attention energies in shape (B,N,A)
        '''
        attr_len = encoder_outputs.size(0) # A
        seq_len = hidden.size(0) # N
        batch_size = encoder_outputs.size(1)

        H = hidden.repeat(attr_len,1,1,1) # [A,N,B,H]
        encoder_outputs = encoder_outputs.repeat(seq_len,1,1,1).transpose(0,1).contiguous() # [N,A,B,K] -> [A,N,B,K]

        attn_energies = F.tanh(self.score(H, encoder_outputs)) # compute attention score [B,N,A]
        return F.softmax(attn_energies, dim=2) # normalize with softmax on A axis

    def score(self, hidden, encoder_outputs): # hidden (A,N,B,H)
        concat = torch.cat([hidden, encoder_outputs], 3)  # => [A,N,B,(H+K)]
        energy = self.attn(concat) # [A,N,B,(H+K)]->[A,N,B,H]
        energy = energy.view(-1, self.hidden_size) # [A*N*B,H]
        v = self.v.unsqueeze(1) # [H,1]
        energy = energy.mm(v) # [A*N*B,H] x [H,1] -> [A*N*B,1]
        att_energies = energy.view(-1, hidden.size(1), hidden.size(2)) # [A,N,B] 
        att_energies = att_energies.transpose(0, 2).contiguous() # [B,N,A]
        return att_energies

class TopicAttnDecoderRNN(nn.Module):
    def __init__(self, topic_embedding, embed_size, hidden_size, attr_size, output_size, n_layers=1, dropout=0.2):
        super(TopicAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.embed_size = embed_size 
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.attr_size = attr_size
        self.n_layers = n_layers
        self.dropout = dropout
        
        # Define layers
        self.topic_embedding = topic_embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size + attr_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Choose attention model
        self.attr_attn = AttributeAttn(hidden_size, attr_size)

    def forward(self, input_seq, last_hidden, encoder_out):
        # Note: we run all steps at one pass
        # Get the embedding of all input words
        '''
        :param input_seq:
            topic input for all time steps, in shape (N=seq_len, B)
        :param last_hidden:
            tuple, last hidden stat of the decoder, in shape (layers, B,H)
        :param encoder_out: 
            encoder outputs from attribute in shape (A=attr_num,B,K=attr_size)
        '''
        
        batch_size = input_seq.size(1) # B
        seq_len = input_seq.size(0) # N
        topic_embedded = self.topic_embedding(input_seq) # [N,B,E]
        topic_embedded = self.embedding_dropout(embedded)

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(topic_embedded, last_hidden)  # rnn_output: [N=seq_len, B, H]  hidden: [n_layer, B, H]

        # Calculate attention
        attn_weights = self.attr_attn(rnn_output, encoder_out) # [N,B,H] x [A,B,K] -> [B,N,A]
        topic_context = attn_weights.bmm(encoder_out.transpose(0, 1).contiguous()) # [B,N,A] x [B,A,K] -> [B,N,K]
        topic_context = topic_context.transpose(0, 1).contiguous()        # [B,N,K] -> [N,B,K]

        tanh_input = torch.cat((rnn_output, topic_context), 2) # [N,B, H+K]
        tanh_output = F.tanh(self.concat(tanh_input))  # [N,B,H]
        
        output = self.out(tanh_output) # [N,B,W=vsize]

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights

