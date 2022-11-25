# National Artificial Intelligence Competition (AI CUP 2021) - Agriculture Articles Label Classification
    
<!-- Badges -->
<p>
  <a href="">
    <img src="https://img.shields.io/badge/contributors-5-yellow" alt="contributors" />
  </a>
  <a href="">
    <img src="https://img.shields.io/badge/last%20update-December%202021-green" alt="last update" />
  </a>
    <a href="">
    <img src="https://img.shields.io/badge/Rank-13th-blue" alt="Rank" />
  </a>
</p>

## Competition information

[AI CUP Competition](https://aidea-web.tw/topic/de144f63-cd15-40b8-81e6-82db5636d598)

Team Member : Alison Chi, Daniel Kong, Hong-Wen Wang, Yung-Yu Shi, Drew Cavicchi\
Final Rank : **13**/118\
Public F-Score : **0.68** (13/118)

## Abstract

This repository contains the codes and data used in the 2021 AI CUP competition. 

##  Introduction

For our final project, we took two main approaches: 

One was the “document retrieval” approach, in which we used cosine or Jaccard similarity to retrieve relevant documents using a variety of embedding techniques including TF-IDF, Doc2Vec, and BERT. 

The second was the binary “document classification” approach, where we trained a variety of BERT models to classify any given document pair as similar or not.

We were ranked number 13 out of 118 teams with an overall F-score of 0.68. 

## Experiments and Results

#### General Preprocessing
We used “Ckip” an open-source library that implements neural Chinese NLP tools, to tokenize all the paragraphs with respect to their context. We then created two versions of tokens, one replacing all the keywords with the synonyms, the other one keeping all the words intact. We used Ckip to tokenize and calculate the total number of tokens. Finally, we constructed the reference table using the training dataset provided by the competition organizer.

---

#### TF-IDF with Distance Measures

1. Cosine Similarity

Using basic TF-IDF with Cosine Similarity yielded our best results. Our first attempts with simple preprocessing and roughly .75% confidence threshold yielded f-scores of around .38. Our major adjustments included making sure key words were included in the corpus, adjusting the weight of key words by adjusting the number of ccurrences per document, and by parameter tuning. We also detected county and location names in the documents and added occurrences using the same method as we did with key words. 
Overall, these techniques improved our score to .5 on the public test data, and .48 on the private test data. We found that the most important factors to improving the score were by far the threshold, number of keyword occurrences added, and number of location occurrences added. After some analysis of results and adjustments of preprocessing, we found our best combination to be a threshold of roughly .7, around 20 additional keyword occurrences, and 8 additional location occurrences.

Token tf-idf mean score
![alt text](https://github.com/bluebird1337/Agriculture-Articles-Label-Classification/blob/main/cosine_similarity.png "Token tf-idf mean score")

2. Jaccard Similarity

We modified some parts of our preprocessing to facilitate the use of Jaccard. Same as before, we removed the punctuation and the stopwords. We then replaced all related synonyms provided by the organizer with the first word in the provided keyword lists. Next, we used Ckip to tokenize and the Jaccard similarity method to compute the similarity between two text documents. Finally, experimented with thresholds to filter out the data we don't want. The best result we got was about 0.42.

3. Optimizing TF-IDF

We took a look at the top 50 TF-IDF features for all documents on average and found something interesting. One particular term, “037”, was among the features returned by the TF-IDF model, despite not being a relevant term for comparison. To remedy this, we could have implemented additional preprocessing to remove numerical elements from the tokens. In addition, we investigated the possibility that some of our long keywords were being split up by the tokenizer. To mitigate this, we constructed a dictionary object containing our keywords with weights assigned. By passing this to the tokenizer, it ensures that the long keywords will stay intact

4. Results Evaluation

In addition, we have also used a confusion matrix to evaluate the quality of the output of a classifier on this dataset. By the way, the TF might be “N/A” instead of 0 because its probability is not useful for our prediction (but the plot package does not allow us to modify this way)

Confusion Matrix
![alt text](https://github.com/bluebird1337/Agriculture-Articles-Label-Classification/blob/main/confusion_matrix.png "Confusion Matrix")

---

#### Doc2Vec

We also tested Doc2Vec using gensim. We trained on the entire testing and training data set, and gathered the “most similar” documents per document using built in eatures with gensim. Interestingly, we found that Doc2Vec required much higher confidence threshold cutoffs than TF-IDF - our best results came in with a 95% cutoff. We attempted hyperparameter tuning here as well, but found that threshold was by far the most important factor in deciding performance. Overall, performance here capped out at roughly a .38 f-score, so we did not pursue further optimization.

---

#### BERT

We chose BERT because it is a state-of-the-art language model and can be easily fine-tuned for downstream tasks. Altogether, we trained 16 BERT models. We used the sentence-transformers library, a pytorch wrapper library developed by the authors of the Sentence-BERT paper [4]. Although it was made for sentences, there was no token length limit besides the 512 model limit itself, so it could work with full documents. The input training data format for BERT required a TSV document with sentence airs and then a label 0 or 1 for not similar and similar respectively. To create our training data, we labelled all similar pairs as 1 and then sampled negative pairs rom the dataset, considering pairs that are not in each other’s relevant document ID lists to be not similar. We tried both balanced (1-1) and imbalanced (1-10, where here were 10 times more negative pairs) classes. We were concerned that the training dataset was too small, so for some models, we did not have a validation set (and ust used the training data as validation), and for some models, we held out a 10% size validation set for validation during training.

#### Result

Model | Training data F-score | Submittsion F-score
--- | --- | ---
bert-base-chinese, batch size 4 | 88% | 65%
macbert, batch size 4 | 71% | 60%











