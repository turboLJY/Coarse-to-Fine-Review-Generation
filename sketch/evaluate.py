import torch
from torch.autograd import Variable
import random
from model import *
from util import *
from config import USE_CUDA, save_dir 
import sys
import os
from masked_cross_entropy import *
import itertools
import random
import math
from tqdm import tqdm
from load import SOS_ID, EOS_ID, PAD_ID
from model import TopicAttnDecoderRNN, SketchAttnDecoderRNN, AttributeEncoder
import pickle
import logging

logging.basicConfig(level=logging.INFO)
USE_CUDA = torch.cuda.is_available()

MAX_TOPIC = 10
MIN_TOPIC = 2

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

def beam_decode(topic_decoder, sketch_decoder, topic_decoder_hidden, sketch_decoder_hidden, topic_encoder_out, sketch_encoder_out, vocab, beam_size, max_length, min_length):
    topic_input = Variable(torch.LongTensor([[SOS_ID]]))
    topic_input = topic_input.cuda() if USE_CUDA else topic_input

    topic_hidden = topic_decoder_hidden 
    sketch_hidden = sketch_decoder_hidden 

    decoded_topics = []
    decoded_sketchs = []

    for ti in range(MAX_TOPIC):
        topic_output, topic_hidden, _ = topic_decoder(topic_input, topic_hidden, topic_encoder_out)
        topv, topi = topic_output.data.topk(5)
        topi = topi.squeeze(0)
        i = 0
        
        # for electronic
        while (i < 4 and topi[0][i] < 3 and len(decoded_topics) < MIN_TOPIC) or (i < 4 and vocab.idx2topic[topi[0][i]] in decoded_topics):
            i += 1

        nti = topi[0][i]
        if nti == EOS_ID:
            decoded_topics.append("<eos>")
            break
        else:
            decoded_topics.append(vocab.idx2topic[nti])
            topic = Variable(torch.LongTensor([[nti]]))
            topic = topic.cuda() if USE_CUDA else topic
            hyps = [Hypothesis(tokens=[SOS_ID], log_probs=[0.0], state=sketch_hidden) for _ in range(beam_size)] 
            results = [] 
            steps = 0
            while steps < max_length and len(results) < beam_size:
                new_hiddens = []
                topk_ids = []
                topk_probs = []
                for hyp in hyps:
                    sketch_input = Variable(torch.LongTensor([[hyp.latest_token]]))
                    sketch_input = sketch_input.cuda() if USE_CUDA else sketch_input
            
                    sketch_hidden = hyp.state 
            
                    sketch_output, sketch_hidden, _ = sketch_decoder(sketch_input, sketch_hidden, topic, sketch_encoder_out)
                    new_hiddens.append(sketch_hidden)
            
                    topv, topi = sketch_output.data.topk(beam_size*2)
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
            best_hyp = hyps_sorted[0]
            sketch_tokens = best_hyp.tokens[1:]
            sketch_tokens = [vocab.idx2sketch[t] for t in sketch_tokens]
            if sketch_tokens[-1] != "<eos>":
                sketch_tokens.append("<eos>")
            decoded_sketchs.extend(sketch_tokens)

            sketch_hidden = best_hyp.state 

        topic_input = Variable(torch.LongTensor([[nti]]))
        topic_input = topic_input.cuda() if USE_CUDA else topic_input

    if decoded_topics[-1] != "<eos>":
        decoded_topics.append("<eos>")

    return decoded_topics, decoded_sketchs 

def decode(topic_decoder, sketch_decoder, topic_decoder_hidden, sketch_decoder_hidden, topic_encoder_out, sketch_encoder_out, vocab, max_length, min_length):

    topic_input = Variable(torch.LongTensor([[SOS_ID]]))
    topic_input = topic_input.cuda() if USE_CUDA else topic_input

    topic_hidden = topic_decoder_hidden 
    sketch_hidden = sketch_decoder_hidden 

    decoded_topics = []
    decoded_sketchs = []

    for ti in range(MAX_TOPIC):
        topic_output, topic_hidden, _ = topic_decoder(topic_input, topic_hidden, topic_encoder_out)
        topv, topi = topic_output.data.topk(4)
        topi = topi.squeeze(0)
        nti = topi[0][0]
        if nti == EOS_ID:
            decoded_topics.append("<eos>")
            break
        else:
            decoded_topics.append(vocab.idx2topic[nti])
            topic = Variable(torch.LongTensor([[nti]]))
            topic = topic.cuda() if USE_CUDA else topic
            sketch_input = Variable(torch.LongTensor([[SOS_ID]]))
            sketch_input = sketch_input.cuda() if USE_CUDA else sketch_input
            for pi in range(MAX_LENGTH):
                sketch_output, sketch_hidden, _ = sketch_decoder(sketch_input, sketch_hidden, topic, sketch_encoder_out)
                topv, topi = sketch_output.data.topk(4)
                topi = topi.squeeze(0)
                npi = topi[0][0]
                if npi == EOS_ID:
                    decoded_sketchs.append("<eos>")
                    break 
                else:
                    decoded_sketchs.append(vocab.idx2sketch[npi])

                sketch_input = Variable(torch.LongTensor([[npi]]))
                sketch_input = sketch_input.cuda() if USE_CUDA else sketch_input

            if decoded_sketchs[-1] != "<eos>":
                decoded_sketchs.append("<eos>")

            sketch_hidden = sketch_hidden

        topic_input = Variable(torch.LongTensor([[nti]]))
        topic_input = topic_input.cuda() if USE_CUDA else topic_input

    if decoded_topics[-1] != "<eos>":
        decoded_topics.append("<eos>")

    return decoded_topics, decoded_sketchs 


def evaluate(topic_encoder, topic_decoder, sketch_encoder, sketch_decoder, vocab, pair, beam_size, max_length, min_length):
    attribute = pair # (user_id, item_id, rating)
    attr_input = Variable(torch.LongTensor([attribute]), volatile=True)
    attr_input = attr_input.cuda() if USE_CUDA else attr_input

    # attribute encoder
    topic_encoder_out, topic_encoder_hidden = topic_encoder(attr_input) 
    sketch_encoder_out, sketch_encoder_hidden = sketch_encoder(attr_input)

    # topic embedding
    topic_decoder_hidden = topic_encoder_hidden[:topic_decoder.n_layers]
    
    # sketch decoder
    sketch_decoder_hidden = sketch_encoder_hidden[:sketch_decoder.n_layers]
    
    if beam_size == 1:
        return decode(topic_decoder, sketch_decoder, topic_decoder_hidden, sketch_decoder_hidden, topic_encoder_out, sketch_encoder_out, vocab, max_length, min_length)
    else:
        return beam_decode(topic_decoder, sketch_decoder, topic_decoder_hidden, sketch_decoder_hidden, topic_encoder_out, sketch_encoder_out, vocab, beam_size, max_length, min_length)

def evaluateRandomly(topic_encoder, topic_decoder, sketch_encoder, sketch_decoder, vocab, user_rdict, item_rdict, pairs, n_pairs, beam_size, max_length, min_length, save_dir):
    path = os.path.join(save_dir, 'decode')
    if not os.path.exists(path):
        os.makedirs(path)
    f1 = open(path + "/decoded.txt", 'w')
    for i in range(n_pairs):
    
        pair = pairs[i]
        
        user = pair[0][0]
        item = pair[0][1]
        rating = pair[0][2]
        attribute = '\t'.join([user_rdict[user], item_rdict[item], str(rating+1)])
        topic = " ".join(pair[1])
        sketchs = []
        for sketch in pair[2]:
            sketchs.append(" ".join(sketch[1:-1]))
        sketchs = "||".join(sketchs)
        print("=============================================================")
        print('Attribute > ', attribute)
        print('Topic > ', topic)
        print('Reference > ', sketchs)

        f1.write('Attribute: ' + attribute + '\n' + 'Topic: ' + topic + '\n' + 'Reference: ' + sketchs + '\n')
        if beam_size >= 1:
            output_topics, output_sketchs = evaluate(topic_encoder, topic_decoder, sketch_encoder, sketch_decoder, vocab, pair[0], beam_size, max_length, min_length)
            topic_sentence = ' '.join(output_topics[:-1])
            sketch_words = []
            for wd in output_sketchs:
                if wd == "</s>":
                    sketch_words.append("||")
                else:
                    sketch_words.append(wd)
            sketch_sentence = ' '.join(sketch_words[:-1])
            print('Generation topic < ', topic_sentence)
            print('Generation sketch < ', sketch_sentence)
            f1.write('Generation topic: ' + topic_sentence + "\n")
            f1.write('Generation sketch: ' + sketch_sentence + "\n")
    f1.close()

def runTest(corpus, n_layers, hidden_size, embed_size, attr_size, attr_num, overall, sk_modelFile, tp_modelFile, beam_size, max_length, min_length, save_dir):

    vocab, train_pairs, valid_pairs, test_pairs = loadPrepareData(corpus, save_dir)
    
    print('Building encoder and decoder ...')

    # topic encoder
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

    topic_encoder = AttributeEncoder(attr_size, attr_num, hidden_size, attr_embeddings, n_layers)

    # topic decoder
    topic_embedding = nn.Embedding(vocab.n_topics, embed_size)   

    if USE_CUDA:
        topic_embedding = topic_embedding.cuda()

    topic_decoder = TopicAttnDecoderRNN(topic_embedding, embed_size, hidden_size, attr_size, vocab.n_topics, n_layers)
    
    checkpoint = torch.load(tp_modelFile)
    topic_encoder.load_state_dict(checkpoint['encoder'])
    topic_decoder.load_state_dict(checkpoint['topic_decoder'])

    # use cuda
    if USE_CUDA:
        topic_encoder = topic_encoder.cuda()
        topic_decoder = topic_decoder.cuda()

    # train mode set to false, effect only on dropout, batchNorm
    topic_encoder.train(False)
    topic_decoder.train(False)

    # sketch encoder
    attr_embeddings = []
    attr_embeddings.append(nn.Embedding(num_user, attr_size))
    attr_embeddings.append(nn.Embedding(num_item, attr_size))
    attr_embeddings.append(nn.Embedding(num_over, attr_size))

    if USE_CUDA:
        for attr_embedding in attr_embeddings:
            attr_embedding = attr_embedding.cuda()

    sketch_encoder = AttributeEncoder(attr_size, attr_num, hidden_size, attr_embeddings, n_layers)

    # sketch decoder
    topic_embedding = nn.Embedding(vocab.n_topics, embed_size)
    sketch_embedding = nn.Embedding(vocab.n_sketchs, embed_size)   

    if USE_CUDA:
        topic_embedding = topic_embedding.cuda()
        sketch_embedding = sketch_embedding.cuda()

    sketch_decoder = SketchAttnDecoderRNN(topic_embedding, sketch_embedding, embed_size, hidden_size, attr_size, vocab.n_sketchs, n_layers)
    
    checkpoint = torch.load(sk_modelFile)
    sketch_encoder.load_state_dict(checkpoint['encoder'])
    sketch_decoder.load_state_dict(checkpoint['sketch_decoder'])

    # use cuda
    if USE_CUDA:
        sketch_encoder = sketch_encoder.cuda()
        sketch_decoder = sketch_decoder.cuda()

    # train mode set to false, effect only on dropout, batchNorm
    sketch_encoder.train(False)
    sketch_decoder.train(False)

    evaluateRandomly(topic_encoder, topic_decoder, sketch_encoder, sketch_decoder, vocab, user_rdict, item_rdict, test_pairs, len(test_pairs), beam_size, max_length, min_length, save_dir)
   