# -*- coding: utf-8 -*-
"""fine_tune_bert.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EBeNXodeIoA5l8NNgtcwJLRwhc3OiWf3
"""

#!pip install sentence_transformers

import os
import pandas as pd
from sentence_transformers import SentenceTransformer, losses, readers, evaluation, InputExample
from torch.utils.data import DataLoader
#from google.colab import files

MODEL_NAME = 'bert-base-chinese'
batch_size = 4

model = SentenceTransformer(MODEL_NAME)

# from google.colab import drive
# drive.mount('/content/drive')

def get_examples(fpath):
  examples = []
  fname = fpath.split('/')[-1].split('.')[-2]
  gu_id = 0
  for line in open(fpath, encoding="utf-8"):
    splits = line.strip().split('\t')
    label = int(splits[2])
    text1 = splits[0]
    text2 = splits[1]
    guid = "%s-%d" % (fname, gu_id)
    gu_id += 1
    examples.append(InputExample(guid=guid, texts=[text1, text2], label=label))
  return examples

train_pair_examples = get_examples('all_pairs_balanced.tsv')

print(train_pair_examples[0])

def get_three_lists_for_evaluator(fpath):
  texts1 = []
  texts2 = []
  labels = []
  fname = fpath.split('/')[-1].split('.')[-2]
  for line in open(fpath, encoding="utf-8"):
    splits = line.strip().split('\t')
    label = int(splits[2])
    text1 = splits[0]
    text2 = splits[1]
    texts1.append(text1)
    texts2.append(text2)
    labels.append(label)
  return texts1, texts2, labels

train_data_loader = DataLoader(train_pair_examples, batch_size=batch_size, shuffle=True)

train_loss = losses.SoftmaxLoss(model=model,
                                sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                num_labels=2)

texts1, texts2, labels = get_three_lists_for_evaluator('all_pairs_balanced.tsv')

evaluator = evaluation.BinaryClassificationEvaluator(batch_size=batch_size, sentences1=texts1, sentences2=texts2, labels=labels)

model_save_path = 'bert_chinese_doc_pair_classifier'

model.fit(train_objectives=[(train_data_loader, train_loss)], epochs=4,
          output_path=model_save_path, evaluator=evaluator, evaluation_steps=500)