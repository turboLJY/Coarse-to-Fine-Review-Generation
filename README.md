# Coarse-to-Fine-Review-Generation
this repository contains the source code for the ACL 2019 paper "[Generating Long and Informative Reviews with Aspect-Aware Coarse-to-Fine Decoding](https://arxiv.org/abs/1906.05667)"

# Requirements:

- Python 3.6
- Pytorch 0.3
- Anaconda3

# Preparing the dataset

- [Amazon Electronic](http://jmcauley.ucsd.edu/data/amazon/links.html)
- [Yelp Restaurant](https://www.yelp.com/dataset/challenge)
- [RateBeer](http://cseweb.ucsd.edu/~jmcauley/datasets.html#multi_aspect)

## topic module preprocess steps

- First, we convert all review texts into lowercase, perform tokenization using NLTK, and split the tokenized review texts into sentences
- Second, we run the TwitterLDA model in the sentences, and tag each sentences.
- Third, we get the topic/aspect sequence about a user-item review, and the top 100 topic words in every topic/aspect.

Finally, we get the files

```
"topic.pkl": topic2idx dictionary, including <sos>, <eos>, <unk>, <pad>, and topic labels.
"topic_rev.pkl": idx2topic dictionary, the reverse of topic2idx
"user.pkl" and "item.pkl": user2idx and item2idx dictionary
```

## sketch module preprocess steps

- First, we count the uni-gram, bi-gram and tri-gram. we get the top 50 uni-gram, 200 bi-gram and 200 tri-gram.
- Second, we run the StanfordPostagger in the tokenized review texts.
- Third, to get the sketch, we keep the words ranked in topic words and n-grams, and replace the rest words with their Part-of-Speech tags.

Finally, we get the files

```
"sketch.pkl": sketch2idx dictionary, including <sos>, <eos>, <pad>, top 50 topic words, n-grams, and Part-of-Speech tags.
"sketch_rev.pkl": idx2sketch dictionary, the reverse of sketch2idx
```

## review module preprocess steps

- we build a dictionary in the tokenized review texts

Finally, we get the files

```
"review2idx.pkl": review2idx dictionary, including the words occuring no less than 5 times.
"idx2review.pkl": idx2review dictionary, the reverse of review2idx.
"aspect_ids.pkl": topic words list, every word is replaced by its idx. the length of list is 100 * topics.
```

The last json file format:
```
Example:
{"asin": "B000M17AVO",
   "reviewerID": "AAXUSC3RGM4ZJ", 
   "overall": 4,
   "topic": "6 1", 
   "topic_tok": ["6", "1"], 
   "sketchText": "if you use PRP$ NN for watching dvds , NN .||the remote is NN of JJ . is VBG a JJ on JJ button .", 
   "reviewText": "if you use your ps3 for watching dvds , divx .||the remote is kind of cluttered . is lacking a direct on off button ."
   }
```

# Training Instruction

```
sh run.sh
```

Because we have the gold standard in every stage, you can train topic, sketch and review module concurrently and save the models in every stage. 

# Testing Instruction

You can test the performance in every stage. You need to be aware that 1) testing in the sketch stage will use the topic model; 2) testing in the review stage will use the topic and sketch model.

# Citation

If this work is useful in your research, please cite our paper.

```
@inproceedings{junyi2019review,
  title={{G}enerating {L}ong and {I}nformative {R}eviews with {A}spect-{A}ware {C}oarse-to-{F}ine {D}ecoding},
  author={Junyi Li, Wayne Xin Zhao, Ji-Rong Wen, and Yang Song},
  booktitle={ACL},
  year={2019}
}
```

