#!/usr/bin/env python3  
#coding: utf-8
"""利用第三方库gensim进行训练得到词向量"""
#词向量模型的简单实现
import json
import jieba
import numpy as np
import gensim
from gensim.models import Word2Vec  #词的向量化
from collections import defaultdict

#训练模型
#corpus: [["cat", "say", "meow"], ["dog", "say", "woof"]]
#dim指定词向量的维度，如100
def train_word2vec_model(corpus, dim):
    model = Word2Vec(corpus, vector_size=dim, sg=1)
    model.save("model.w2v")
    return model

#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def main():
    sentences = []
    with open("F:/bdou/ppt课件/第四周/第四周 词向量及应用/课件/week4 词向量及文本向量/week4 词向量及文本向量/corpus.txt", encoding="utf8") as f:
        for line in f:
            """jieba分词方法"""
            sentences.append(jieba.lcut(line))
    model = train_word2vec_model(sentences, 100)
    return

if __name__ == "__main__":
    main()
    model = load_word2vec_model("model.w2v")
    while True:
        string = input("input:")
        try:
            print(model.wv.most_similar(string))  #查找最相近的词
        except KeyError:
            print("输入词不存在")