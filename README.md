# Persian-Sentiment-Analysis
# Introduction
Sentiment analysis is one of the challenging tasks in the Natural Language Processing area, especially when it comes to applying to the Persian language. An evolution in NLP results occurred after introducing BERT by Google.
In part one we talk about the dataset that we used. Second part describes development details.
# Dataset
We have collected data from some websites including: https://snappfood.ir, https://www.digikala.com/ and https://taaghche.com. The number of samples is 196,387 which contains 124,010 positive and 72,377 negative samples. After collecting the dataset we tried to clean the data samples and correct obviously misclassified ones.
# Implementation details
Fine tuning Bert Model mainly consists of three parts. First step is designing and attaching a head to the base model. Head part consists of some layers which should be designed based on the task (classification here). Second step includes preprocessing the corpus and converting text strings into tokens. We used a pre-trained Bert tokenizer for this aim. In the cleaning phase, we should consider the problem needs. For example in the sentiment analysis task we are not allowed to remove negative stop words, because they have polarity meaning in that context.
A flask based API also has been implemented which works with POST method and returns the polarity score for both classes (positive and negative)
# Results
In this section we present the results. Please note that before introducing transformer base models (like Bert) achieving the accuracy about 70% was hard to reach.
AUC	Accuracy	F score	Recall	Precision	 
93.74	86.48	86.59	86.48	86.92	Result in %
