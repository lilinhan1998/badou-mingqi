#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
#修改word2vec_kmeans文件,使得输出类别按照类内平均距离排序

import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

#输入模型文件路径
#加载训练好的模型
#文本转词向量模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

#返回分词文本
def load_sentence(path):
    #无序不重复的元素集
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            #每个句子的分词
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  #sentence是分好词的，空格分开
        #创建词向量维度大小的向量
        vector = np.zeros(model.vector_size)
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)  #文本标题向量

#单句子向量化
def single_sentence_to_vector(sentence,model):
    words=sentence.split()
    vector=np.zeros(model.vector_size)
    for word in words:
        try:
            vector += model.wv[word]
        except KeyError:
            #部分词在训练中未出现，用全0向量代替
            vector += np.zeros(model.vector_size)
    vector=vector/len(words)
    return np.array(vector)

def kmeans_cal(sentences,vectors):
    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量(样本数量的开根号)
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算
    return kmeans

def vectors_with_label(sentences,kmeans):
    sentence_label_dict = defaultdict(list)
    #defaultdict函数在字典key不存在时，返回默认值，List即返回空列表
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(single_sentence_to_vector(sentence,model))  #同标签的文本向量放到一起
    return sentence_label_dict

def cal_average_distances(sentence_label_dict,centers):
    distances={}
    for label, vectors in sentence_label_dict.items():
        #返回可遍历的(键, 值) 元组数组
        distance=0
        for i in range(len(vectors)):
            distance += np.sqrt(0.01*np.abs((np.sum(vectors[i]-centers))))/len(vectors)
        distances[label]=distance
    tuple1=zip(distances.values(),distances.keys())
    distances=list(sorted(tuple1))
    return distances
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
    #         #将前面的字符替换成后面的，旧的换成新的
    #         print(sentences[i].replace(" ", ""))
    #     print("---------")

if __name__ == "__main__":
    model = load_word2vec_model("model.w2v") #加载词向量模型
    path='F:/bdou/ppt课件/第四周/第四周 词向量及应用/课件/week4 词向量及文本向量/week4 词向量及文本向量/titles.txt'
    sentences = load_sentence(path)  #加载所有标题，返回分词文本
    vectors = sentences_to_vectors(sentences, model)  #将所有标题向量化(1796,100)
    kmeans=kmeans_cal(sentences,vectors)
    centers=kmeans.cluster_centers_
    sentence_label_dict=vectors_with_label(sentences,kmeans)
    distances=cal_average_distances(sentence_label_dict,centers)
    print('(类内平均间距，簇编号)：')
    print(distances)
    #distances=kmeans.inertia_