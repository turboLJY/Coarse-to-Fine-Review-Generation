import torch
from torch.autograd import Variable
import random
from model import *
from util import *
import sys
import os
from masked_cross_entropy import *
import itertools
import random
import math
from tqdm import tqdm
from load import SOS_ID, EOS_ID, PAD_ID
from model import TopicAttnDecoderRNN, AttributeEncoder
import pickle
import logging
logging.basicConfig(level=logging.INFO)
USE_CUDA = torch.cuda.is_available()

class Hypothesis(object):
    def __init__(self, tokens, log_probs, state):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state

    def extend(self, token, log_prob, state):
        return Hypothesis(tokens = self.tokens + [token],
                        log_probs = self.log_probs + [log_prob],
                        state = state)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property 
    def log_prob(self):
        return sum(self.log_probs)

    @property 
    def avg_log_prob(self):
        return self.log_prob / len(self.tokens)
 
def sort_hyps(hyps):
    return sorted(hyps, key=lambda h: h.log_prob, reverse=True)

def beam_decode(topic_decoder, decoder_hidden, encoder_out, vocab, beam_size, max_length, min_length):
    hyps = [Hypothesis(tokens=[SOS_ID], log_probs=[0.0], state=decoder_hidden) for _ in range(beam_size)] 
    results = [] 
    steps = 0
    while steps < max_length and len(results) < beam_size:
        new_hiddens = []
        topk_ids = []
        topk_probs = []
        for hyp in hyps:
            decoder_input = Variable(torch.LongTensor([[hyp.latest_token]]))
            decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input
            
            decoder_hidden = hyp.state 
            
            decoder_output, decoder_hidden, _ = topic_decoder(decoder_input, decoder_hidden, encoder_out)
            new_hiddens.append(decoder_hidden)
            
            topv, topi = decoder_output.data.topk(beam_size*2)
            topv = topv.squeeze(0)
            topi = topi.squeeze(0)

            topk_ids.extend(topi)
            topk_probs.extend(torch.log(topv))

        all_hyps = []   
        num_orig_hyps = 1 if steps == 0 else len(hyps) 
        for i in range(num_orig_hyps):
            h, new_hidden = hyps[i], new_hiddens[i] 
            for j in range(beam_size*2):  
                new_hyp = h.extend(token=topk_ids[i][j], log_prob=topk_probs[i][j], state=new_hidden)
                all_hyps.append(new_hyp)

        hyps = []
        for h in sort_hyps(all_hyps): 
            if h.latest_token == EOS_ID: 
                if steps >= min_length:
                    results.append(h)
            else:
                hyps.append(h)
            if len(hyps) == beam_size or len(results) == beam_size:
                break

        steps += 1
        
    if len(results)==0: 
        results = hyps

    hyps_sorted = sort_hyps(results)

    return hyps_sorted[0]

def decode(topic_decoder, decoder_hidden, encoder_out, vocab, max_length, min_length):

    decoder_input = Variable(torch.LongTensor([[SOS_ID]]))
    decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    decoded_words = []

    for di in range(max_length):
        decoder_output, decoder_hidden, _ = topic_decoder(decoder_input, decoder_hidden, encoder_out)

        topv, topi = decoder_output.data.topk(4)
        topi = topi.squeeze(0)
        topv = topv.squeeze(0)
        ni = topi[0][0]
        if ni == EOS_ID:
            decoded_words.append('<eos>')
            break
        else:
            decoded_words.append(vocab.idx2topic[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    if decoded_words != '<eos>':
        decoded_words.append('<eos>')

    return decoded_words


def evaluate(encoder, topic_decoder, vocab, pair, beam_size, max_length, min_length):
    sentence = pair # (user_id, item_id, rating)
    attr_input = Variable(torch.LongTensor([sentence]), volatile=True)
    attr_input = attr_input.cuda() if USE_CUDA else attr_input

    encoder_out, encoder_hidden = encoder(attr_input) 
    decoder_hidden = encoder_hidden[:topic_decoder.n_layers]
    
    if beam_size == 1:
        return decode(topic_decoder, decoder_hidden, encoder_out, vocab, max_length, min_length)
    else:
        return beam_decode(topic_decoder, decoder_hidden, encoder_out, vocab, beam_size, max_length, min_length)


def evaluateRandomly(encoder, topic_decoder, vocab, user_rdict, item_rdict, pairs, n_pairs, beam_size, max_length, min_length, save_dir):

    path = os.path.join(save_dir, 'decode')
    if not os.path.exists(path):
        os.makedirs(path)
    f1 = open(path + "/decoded.txt", 'w')
    for i in range(n_pairs):
    
        pair = pairs[i]
        
        user = pair[0][0]
        item = pair[0][1]
        rating = pair[0][2]
        topic = ' '.join(pair[1][1:-1])
        print("=============================================================")
        print('Attribute > ', '\t'.join([user_rdict[user], item_rdict[item], str(rating+1)]))
        print('Reference > ', topic)

        f1.write('Context: ' + '\t'.join([user_rdict[user], item_rdict[item], str(rating+1)]) + '\n' + 'Reference: ' + topic + '\n')
        if beam_size == 1:
            output_words = evaluate(encoder, topic_decoder, vocab, pair[0], beam_size, max_length, min_length)
            output_sentence = ' '.join(output_words[:-1])
            print('<', output_sentence)
            f1.write('Generation: ' + output_sentence + "\n")
        else:
            best_hyp = evaluate(encoder, topic_decoder, vocab, pair[0], beam_size, max_length, min_length)
            output_idx = [int(t) for t in best_hyp.tokens]
            output_words = [vocab.idx2topic[idx] for idx in output_idx]
            if output_words[-1] != '<eos>':
                output_words.append('<eos>')
            output_sentence = ' '.join(output_words[1:-1])
            f1.write('Generation: ' + output_sentence + '\n')
            print("Generation > ", output_sentence)
    f1.close()

def runTest(corpus, n_layers, hidden_size, embed_size, attr_size, attr_num, overall, modelFile, beam_size, max_length, min_length, save_dir):

    vocab, train_pairs, valid_pairs, test_pairs = loadPrepareData(corpus, save_dir)
    
    print('Building encoder and decoder ...')
    with open(os.path.join(save_dir, 'user_rev.pkl'), 'rb') as fp:
        user_rdict = pickle.load(fp)
    with open(os.path.join(save_dir, 'item_rev.pkl'), 'rb') as fp:
        item_rdict = pickle.load(fp)
        
    num_user = len(user_rdict)
    num_item = len(item_rdict)
    num_over = overall 
    
    attr_embeddings = []
    attr_embeddings.append(nn.Embedding(num_user, attr_size))
    attr_embeddings.append(nn.Embedding(num_item, attr_size))
    attr_embeddings.append(nn.Embedding(num_over, attr_size))
    
    if USE_CUDA:
        for attr_embedding in attr_embeddings:
            attr_embedding = attr_embedding.cuda()

    encoder = AttributeEncoder(attr_size, attr_num, hidden_size, attr_embeddings, n_layers)

    topic_embedding = nn.Embedding(vocab.n_topics, embed_size)   

    if USE_CUDA:
        topic_embedding = topic_embedding.cuda()

    topic_decoder = TopicAttnDecoderRNN(topic_embedding, embed_size, hidden_size, attr_size, vocab.n_topics, n_layers)
    
    checkpoint = torch.load(modelFile)
    encoder.load_state_dict(checkpoint['encoder'])
    topic_decoder.load_state_dict(checkpoint['topic_decoder'])

    # use cuda
    if USE_CUDA:
        encoder = encoder.cuda()
        topic_decoder = topic_decoder.cuda()

    # train mode set to false, effect only on dropout, batchNorm
    encoder.train(False)
    topic_decoder.train(False)

    evaluateRandomly(encoder, topic_decoder, vocab, user_rdict, item_rdict, test_pairs, len(test_pairs), beam_size, max_length, min_length, save_dir)
   