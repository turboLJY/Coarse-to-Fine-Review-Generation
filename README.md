# Coarse-to-Fine-Review-Generation
this repository contains the source code for the ACL 2019 paper "[Generating Long and Informative Reviews with Aspect-Aware Coarse-to-Fine Decoding](https://arxiv.org/abs/1906.05667)"

# Requirements:

- Python 3.6
- Pytorch 0.3
- Anaconda3

# Preparing the dataset

- Amazon Electronic
- Yelp Restaurant
- RateBeer

json file format:
```
Example:
{"asin": "B000M17AVO",
   "reviewerID": "AAXUSC3RGM4ZJ", 
   "overall": 4,
   "topic": "6 1", 
   "topic_tok": ["6", "1"], 
   "sketchText": "if you use PRP$ NN for watching dvds , NN .||the remote is NN of JJ . is VBG a JJ on JJ button .", 
   "reviewText": "if you use your ps3 for watching dvds , divx .||the remote is kind of cluttered . is lacking a direct on off button ."}
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
@inproceedings{koncel2019text,
  title={{G}enerating {L}ong and {I}nformative {R}eviews with {A}spect-{A}ware {C}oarse-to-{F}ine {D}ecoding},
  author={Junyi Li, Wayne Xin Zhao, Ji-Rong Wen, and Yang Song},
  booktitle={ACL},
  year={2019}
}
```

