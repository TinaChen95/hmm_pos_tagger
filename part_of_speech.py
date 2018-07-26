#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/25 13:24
# @Author  : Ting
# 完成词性标注任务
from pos_tagger import Tagger
from nltk.corpus import brown


if __name__ == "__main__":
    tagger = Tagger(corpus=brown.tagged_sents(), n=3)
    print(tagger.tag('Nice to meet you!'))
