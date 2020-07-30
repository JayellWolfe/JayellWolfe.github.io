---
layout: post
title: Naive Bayes Classifier From Scratch
tags: [Naive Bayes, model, classifier, python]
comments: true
---

# What is Naive Bayes ?? 

Naive Bayes Classifier Algorithm is a family of probabilistic classification algorithms based on the [Bayes' Theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem) with the 'naive' assumption of conditional independence between every pair of a feature. 

![NaiveBayes](/img/Bayes_rule-300x172.jpg)

Where: 
P(c|x) is the posterior probability of class (c, target) given predictor (x, attributes).
P(c) is the prior probability of class.
P(x|c) is the likelihood which is the probability of predictor given class.
P(x) is the prior probability of predictor.

There are three types of Naive Bayes Algorithms: 

1. Gaussian Naive Bayes 
    This algorithm follows the Gaussian Normal distribution and supports continuous data with the assumption that the continous values associated with each class are distributed according to a normal (or Gaussian) distribution. 
    
2. Multinominal Naive Bayes 
    This algorithm is suitable for classification with dicrete features (word counts for text classicification for example) it normally requires integer feature counts, however fractional counts such as tf-idf may work as well. 
    
3. Bernoulli Naive Bayes 
    This algorithm is similar to multinominal naive bayes however, in a Bernoulli Naive Bayes, the predictors are boolean variables (yes or no) rather than integers. 
    
    
  
Main advantages of Naive Bayes are that : 

1. fast and easy to understand 
2. not prone to overfitting
3. performs well with small amounts of data
4. when the assumption holds to be true, it performs better compared to other models such as logisitic regression

Some disadvantages of Navie Bayes are that:

1. does not work very well when the number of featrues is very high 
2. the assumption that input features are independent of each other may not always hold to be true
3. important imformation may be lost while resampling the continous variables
