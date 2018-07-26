#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/25 16:01
# @Author  : Ting


def ngram(sentences, n):
    # sentences is list of segmented words
    # e.g. [['Hello', 'World'],[...]]

    counter = dict()
    for sent in sentences:
        sent = ['START']*(n-1) + sent + ['END']*(n-1)
        for i in range(len(sent)-n):
            condition = tuple(w for w in sent[i:i+n-1])
            if condition not in counter:
                counter[condition] = dict()
            if sent[i+n-1] not in counter[condition]:
                counter[condition][sent[i + n - 1]] = 1.0
            else:
                counter[condition][sent[i + n - 1]] += 1.0

    for key in counter:
        total = sum(counter[key].values())
        for value in counter[key]:
            counter[key][value] = counter[key][value]/total

    return counter
