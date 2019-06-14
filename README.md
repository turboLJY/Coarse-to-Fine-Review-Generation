# Coarse-to-Fine-Review-Generation
this repository contains the source code for the ACL 2019 paper "[Generating Long and Informative Reviews with Aspect-Aware Coarse-to-Fine Decoding](https://arxiv.org/pdf/1906.05667.pdf)"

# Requirements:

- Python 3.6
- Pytorch 0.3
- Anaconda3

# Preparing the dataset

- Amazon Electronic
- Yelp
- RateBeer

json file format:
```
Example:
{"asin": "B000M17AVO",
   "reviewerID": "AAXUSC3RGM4ZJ", 
   "overall": 4,
   "topic": "6 1", 
   "topic_tok": \["6", "1"\], 
   "sketchText": "if you use PRP$ NN for watching dvds , NN .||the remote is NN of JJ . is VBG a JJ on JJ button .", 
   "reviewText": "if you use your ps3 for watching dvds , divx .||the remote is kind of cluttered . is lacking a direct on off button ."}
```

# How it works

```
sh run.sh
```

**First**, train topic module and save the topic model; **Second**, load the saved topic model to train sketch module and save the sketch model; **Finally**, load the saved topic and sketch model to train review module and save the review model.
