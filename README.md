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

json data file format:
> Example:
{"asin": "B000M17AVO", <br>
   "reviewerID": "AAXUSC3RGM4ZJ", <br>
   "overall": 4, <br>
   "topic": "6 1", <br>
   "topic_tok": \["6", "1"\], <br>
   "sketchText": "if you use PRP$ NN for watching dvds , NN .||the remote is NN of JJ . is VBG a JJ on JJ button .", <br>
   "reviewText": "if you use your ps3 for watching dvds , divx .||the remote is kind of cluttered . is lacking a direct on off button ."}
