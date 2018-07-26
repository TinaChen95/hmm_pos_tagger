#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/25 16:14
# @Author  : Ting

# 定义 词性标注模型
from hmm import *
from nltk import ConditionalFreqDist, ConditionalProbDist, MLEProbDist
from ngram import ngram


class Tagger(HMM):
    def __init__(self, corpus, n):
        # corpus, 训练标注器的语料， 格式为 [[('Hello', 'NNP'), ('world', 'NN'), ('!', '.')], [...], ...]
        # n - 语言模型 n-gram 中的 n

        # 定义词性标注任务
        # 1. transition 为 n-gram 模型
        # 2. emission 为 P( pos |Word )
        # 3. initial distribution 为 P('START') = 1.0

        # 预处理词库，给每句话加上开始和结束符号
        brown_tags_words = []
        for sent in corpus:
            brown_tags_words.append(('START', 'START'))
            brown_tags_words.extend([(tag[:2], word) for word, tag in sent])
            brown_tags_words.append(('END', 'END'))

        # 从语料集获得 emission - 统计条件概率
        cfd_tagwords = ConditionalFreqDist(brown_tags_words)
        # P(W = word, condition = pos)
        cpd_tagwords = ConditionalProbDist(cfd_tagwords, MLEProbDist)
        emission = {tag: {word: cpd_tagwords[tag].prob(word) for word in cfd_tagwords[tag]} for tag in cpd_tagwords}

        # 从语料集获得 transition - 调用 n-gram 模型
        tags = [[tag for _, tag in sent] for sent in corpus]
        transition = Transition(ngram(tags, n))

        # 定义 initial distribution - 以 START 为句首， 概率为 1
        initial_distribution = {('START',): 1.0}

        # 定义 词性标注器
        HMM.__init__(self, initial_distribution, transition, emission, n)

    def tag(self, sentence):
        # 为句子添加开始符合和结束符号
        sentence = ['START'] * (self.length-1) + ['END'] * (self.length-1)
        return self.viterbi(sentence)
