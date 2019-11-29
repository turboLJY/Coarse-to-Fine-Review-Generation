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
from load import SOS_ID, EOS_ID, PAD_ID
from model import ReviewAttnDecoderRNN, TopicAttnDecoderRNN, SketchAttnDecoderRNN, AttributeEncoder
import pickle
import logging

logging.basicConfig(level=logging.INFO)
USE_CUDA = torch.cuda.is_available()

MAX_TOPIC = 10
MIN_TOPIC = 2
pos = ['NN', 'JJ', 'JJS', 'VBP', 'NNS', 'MD', 'VB', 'IN', 'VBN', 'VBG', 'VBD', 'WDT', 'RB', 'VBZ', 'WP', 'PRP$', 'DT',
         'PRP', 'CD', 'JJR', 'RP', 'WRB', 'CC', 'RBR', 'FW', 'NNP', 'PDT', 'UH', 'WP$', 'RBS', 'TO', 'SYM', 'LS', 'NNPS', 'EX']


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

def beam_decode(topic_decoder, sketch_decoder, birnn_encoder, review_decoder, topic_decoder_hidden, sketch_decoder_hidden, review_decoder_hidden, topic_encoder_out, sketch_encoder_out, review_encoder_out, vocab, beam_size):
    topic_input = Variable(torch.LongTensor([[SOS_ID]]))
    topic_input = topic_input.cuda() if USE_CUDA else topic_input

    topic_hidden = topic_decoder_hidden 
    sketch_hidden = sketch_decoder_hidden 
    review_hidden = review_decoder_hidden 

    decoded_topics = []
    decoded_sketchs = []
    decoded_reviews = []

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

            sketch_tokens = best_hyp.tokens
            if sketch_tokens[-1] != EOS_ID:
                sketch_tokens.append(EOS_ID)

            decoded_sketchs.extend([vocab.idx2sketch[t] for t in sketch_tokens[1:]])

            sketch_hidden = best_hyp.state 

            sketch_out = [[t] for t in sketch_tokens[1:]]

            sketch_rnn = Variable(torch.LongTensor(sketch_out))
            sketch_rnn = sketch_rnn.cuda() if USE_CUDA else sketch_rnn
            birnn_encoder_output, birnn_encoder_hidden = birnn_encoder(sketch_rnn)

            seq_len = len(sketch_out)
            review = []
            topic = Variable(torch.LongTensor([[nti]]))
            topic = topic.cuda() if USE_CUDA else topic
            review_input = Variable(torch.LongTensor([[SOS_ID]]))
            review_input = review_input.cuda() if USE_CUDA else review_input
            for i in range(seq_len):
                sketch = Variable(torch.LongTensor([sketch_out[i]]))
                sketch = sketch.cuda() if USE_CUDA else sketch
                rnn_output = birnn_encoder_output[i].unsqueeze(0)
                rnn_output = rnn_output.cuda() if USE_CUDA else rnn_output 
                review_output, review_hidden, _ = review_decoder(review_input, review_hidden, rnn_output, topic, sketch, review_encoder_out)
                if vocab.idx2sketch[sketch_out[i][0]] in pos:
                    topv, topi = review_output.data.topk(4)
                    topv = topv.squeeze(0)
                    topi = topi.squeeze(0)
                    nwi = topi[0][0]
                    review.append(vocab.idx2word[nwi])
                    review_input = Variable(torch.LongTensor([[nwi]]))
                    review_input = review_input.cuda() if USE_CUDA else review_input
                elif vocab.idx2sketch[sketch_out[i][0]] != EOS_ID:
                    review.append(vocab.idx2sketch[sketch_out[i][0]])
                    review_input = Variable(torch.LongTensor([[sketch_out[i][0]]]))
                    review_input = review_input.cuda() if USE_CUDA else review_input
                else:
                    review.append("<eos>")
                    break 
            decoded_reviews.extend(review)

            review_hidden = review_hidden 

        topic_input = Variable(torch.LongTensor([[nti]]))
        topic_input = topic_input.cuda() if USE_CUDA else topic_input

    if decoded_topics[-1] != "<eos>":
        decoded_topics.append("<eos>")

    return decoded_topics, decoded_sketchs, decoded_reviews 

def decode(topic_decoder, sketch_decoder, birnn_encoder, review_decoder, topic_decoder_hidden, sketch_decoder_hidden, review_decoder_hidden, topic_encoder_out, sketch_encoder_out, review_encoder_out, vocab, max_length, min_length):

    topic_input = Variable(torch.LongTensor([[SOS_ID]]))
    topic_input = topic_input.cuda() if USE_CUDA else topic_input

    topic_hidden = topic_decoder_hidden 
    sketch_hidden = sketch_decoder_hidden 
    review_hidden = review_decoder_hidden 

    decoded_topics = []
    decoded_sketchs = []
    decoded_reviews = []

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

            sketch_out = []
            for pi in range(max_length):
                sketch_output, sketch_hidden, _ = sketch_decoder(sketch_input, sketch_hidden, topic, sketch_encoder_out)
                topv, topi = sketch_output.data.topk(4)
                topi = topi.squeeze(0)
                npi = topi[0][0]
                if npi == EOS_ID:
                    sketch_out.append([EOS_ID])
                    decoded_sketchs.append("<eos>")
                    break 
                else:
                    sketch_out.append([npi])
                    decoded_sketchs.append(vocab.idx2sketch[npi])

                sketch_input = Variable(torch.LongTensor([[npi]]))
                sketch_input = sketch_input.cuda() if USE_CUDA else sketch_input

            if decoded_sketchs[-1] != "<eos>":
                decoded_sketchs.append("<eos>")

            sketch_hidden = sketch_hidden

            sketch_rnn = Variable(torch.LongTensor(sketch_out))
            sketch_rnn = sketch_rnn.cuda() if USE_CUDA else sketch_rnn
            birnn_encoder_output, birnn_encoder_hidden = birnn_encoder(sketch_rnn)

            seq_len = len(sketch_out)
            review = []
            topic = Variable(torch.LongTensor([[nti]]))
            topic = topic.cuda() if USE_CUDA else topic
            review_input = Variable(torch.LongTensor([[SOS_ID]]))
            review_input = review_input.cuda() if USE_CUDA else review_input
            for i in range(seq_len):
                sketch = Variable(torch.LongTensor([sketch_out[i]]))
                sketch = sketch.cuda() if USE_CUDA else sketch
                rnn_output = Variable(torch.FloatTensor([birnn_encoder_output[i]]))
                rnn_output = rnn_output.cuda() if USE_CUDA else rnn_output 
                review_output, review_hidden, _ = review_decoder(review_input, review_hidden, rnn_output, topic, sketch, review_encoder_out)
                if vocab.idx2sketch[sketch_out[i][0]] in pos:
                    topv, topi = review_output.data.topk(4)
                    topv = topv.squeeze(0)
                    topi = topi.squeeze(0)
                    nwi = topi[0][0]
                    review.append(vocab.idx2word[nwi])
                    review_input = Variable(torch.LongTensor([[nwi]]))
                    review_input = review_input.cuda() if USE_CUDA else review_input
                elif vocab.idx2sketch[sketch_out[i][0]] != EOS_ID:
                    review.append(vocab.idx2sketch[sketch_out[i][0]])
                    review_input = Variable(torch.LongTensor([[sketch_out[i+1][0]]]))
                    review_input = review_input.cuda() if USE_CUDA else review_input
                else:
                    review.append("<eos>")
                    break 
            decoded_reviews.extend(review)

            review_hidden = review_hidden 

        topic_input = Variable(torch.LongTensor([[nti]]))
        topic_input = topic_input.cuda() if USE_CUDA else topic_input

    if decoded_topics[-1] != "<eos>":
        decoded_topics.append("<eos>")

    return decoded_topics, decoded_sketchs, decoded_reviews 


def evaluate(topic_encoder, topic_decoder, sketch_encoder, sketch_decoder, review_encoder, birnn_encoder, review_decoder, vocab, pair, beam_size, max_length, min_length):
    attribute = pair # (user_id, item_id, rating)
    attr_input = Variable(torch.LongTensor([attribute]), volatile=True)
    attr_input = attr_input.cuda() if USE_CUDA else attr_input

    # attribute encoder
    topic_encoder_out, topic_encoder_hidden = topic_encoder(attr_input) 
    sketch_encoder_out, sketch_encoder_hidden = sketch_encoder(attr_input)
    review_encoder_out, review_encoder_hidden = review_encoder(attr_input)

    # topic embedding
    topic_decoder_hidden = topic_encoder_hidden[:topic_decoder.n_layers]
    
    # sketch decoder
    sketch_decoder_hidden = sketch_encoder_hidden[:sketch_decoder.n_layers]
    
    # review decoder
    review_decoder_hidden = review_encoder_hidden[:review_decoder.n_layers]
    
    if beam_size == 1:
        return decode(topic_decoder, sketch_decoder, birnn_encoder, review_decoder, topic_decoder_hidden, 
                sketch_decoder_hidden, review_decoder_hidden, topic_encoder_out, sketch_encoder_out, review_encoder_out, vocab, max_length, min_length)
    else:
        return beam_decode(topic_decoder, sketch_decoder, birnn_encoder, review_decoder, topic_decoder_hidden, 
                sketch_decoder_hidden, review_decoder_hidden, topic_encoder_out, sketch_encoder_out, review_encoder_out, vocab, beam_size, max_length, min_length)


def evaluateRandomly(topic_encoder, topic_decoder, sketch_encoder, sketch_decoder, review_encoder, birnn_encoder, review_decoder, vocab, user_rdict, item_rdict, pairs, n_pairs, beam_size, max_length, min_length, save_dir):
    path = os.path.join(save_dir, 'decode')
    if not os.path.exists(path):
        os.makedirs(path)
    f1 = open(path + "/decoded.txt", 'w')
    for i in range(n):
    
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
        reviews = []
        for review in pair[3]:
            reviews.append(" ".join(review[1:-1]))
        reviews = "||".join(reviews)
        print("=============================================================")
        print('Attribute > ', attribute)
        print('Topic > ', topic)
        print('sketch > ', sketchs)
        print('Review > ', reviews)

        f1.write('Attribute: ' + attribute + '\n' + 'Topic: ' + topic + '\n' + 'sketch: ' + sketchs + '\n' + 'Review: ' + reviews + '\n')
        if beam_size >= 1:
            output_topics, output_sketchs, output_reviews = evaluate(topic_encoder, topic_decoder, sketch_encoder, sketch_decoder, review_encoder, birnn_encoder, review_decoder, vocab, pair[0], beam_size, max_length, min_length)
            topic_sentence = ' '.join(output_topics[:-1])
            sketch_words = []
            for wd in output_sketchs:
                if wd == "<eos>":
                    sketch_words.append("||")
                else:
                    sketch_words.append(wd)
            sketch_sentence = ' '.join(sketch_words[:-1])
            review_words = []
            for wd in output_reviews:
                if wd == "<eos>":
                    review_words.append("||")
                elif "_" in wd:
                    review_words.extend(wd.split("_"))
                else:
                    review_words.append(wd)
            review_sentence = ' '.join(review_words[:-1])
            print('Generation topic < ', topic_sentence)
            print('Generation sketch < ', sketch_sentence)
            print('Generation review < ', review_sentence)
            f1.write('Generation topic: ' + topic_sentence + "\n")
            f1.write('Generation sketch: ' + sketch_sentence + "\n")
            f1.write('Generation review: ' + review_sentence + "\n")
    f1.close()

def runTest(corpus, n_layers, hidden_size, embed_size, attr_size, attr_num, overall, rv_modelFile, sk_modelFile, tp_modelFile, beam_size, max_length, min_length, save_dir):

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

    # review encoder
    attr_embeddings = []
    attr_embeddings.append(nn.Embedding(num_user, attr_size))
    attr_embeddings.append(nn.Embedding(num_item, attr_size))
    attr_embeddings.append(nn.Embedding(num_over, attr_size))
    if USE_CUDA:
        for attr_embedding in attr_embeddings:
            attr_embedding = attr_embedding.cuda()

    review_encoder = AttributeEncoder(attr_size, attr_num, hidden_size, attr_embeddings, n_layers)

    # birnn encoder
    sketch_embedding = nn.Embedding(vocab.n_sketchs, embed_size)
    if USE_CUDA:
        sketch_embedding = sketch_embedding.cuda()

    birnn_encoder = EncoderRNN(embed_size, hidden_size, sketch_embedding, n_layers)

    # review decoder
    topic_embedding = nn.Embedding(vocab.n_topics, embed_size)
    sketch_embedding = nn.Embedding(vocab.n_sketchs, embed_size)
    word_embedding = nn.Embedding(vocab.n_words, embed_size)

    with open(os.path.join(save_dir, 'aspect_ids.pkl'), 'rb') as fp:
        ids = pickle.load(fp)

    aspect_ids = nn.Embedding(vocab.n_topics-3, 100)
    aspect_ids.weight.data.copy_(torch.from_numpy(np.array(ids)))

    if USE_CUDA:
        topic_embedding = topic_embedding.cuda()
        sketch_embedding = sketch_embedding.cuda()
        word_embedding = word_embedding.cuda()

    review_decoder = ReviewAttnDecoderRNN(topic_embedding, sketch_embedding, word_embedding, embed_size, hidden_size, attr_size, vocab.n_words, aspect_ids, n_layers)

    checkpoint = torch.load(rv_modelFile)
    review_encoder.load_state_dict(checkpoint['encoder'])
    birnn_encoder.load_state_dict(checkpoint['birnn_encoder'])
    review_decoder.load_state_dict(checkpoint['review_decoder'])

    # use cuda
    if USE_CUDA:
        review_encoder = review_encoder.cuda()
        birnn_encoder = birnn_encoder.cuda()
        review_decoder = review_decoder.cuda()

    # train mode set to false, effect only on dropout, batchNorm
    review_encoder.train(False)
    birnn_encoder.train(False)
    review_decoder.train(False)

    evaluateRandomly(topic_encoder, topic_decoder, sketch_encoder, sketch_decoder, review_encoder, birnn_encoder, review_decoder, 
                        vocab, user_rdict, item_rdict, test_pairs, len(test_pairs), beam_size, max_length, min_length, save_dir)
   
