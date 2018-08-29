# -*- coding: utf-8 -*- 
"""
Baseline methods

 Xia Cui
 08/2018
"""

import sys
import glob
import pickle,os
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from operator import add

def read_labeled(domain):
    input_file = open('../data/%s.labeled'%domain,'r')
    lines = input_file.readlines()
    labels = [int(line.strip().split()[0]) for line in lines]
    contents = [line.strip().split()[1:] for line in lines]
    contents = [[word.lower() for word in line] for line in contents]
    return labels, contents



def load_filtered_glove(filtered_features,gloveFile):
    print "Loading GloVe Model"
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        if word in filtered_features:     
            model[word] = embedding
        # if word.replace('.','__') in filtered_features:
        #     model[word.replace('.','__')] = embedding
    print "After filtering, ",len(model)," words loaded!"
    return model

def collect_all():
    domains = ['reddit','twitter']
    all_data = list()
    for domain in domains:
        all_data += read_labeled(domain)[1]
    all_features = set(x for reivew in all_data for x in reivew)
    print len(all_features)
    return all_features

def save_new_glove_model():
    all_features = collect_all()
    path = "../../pivot-selection/data/glove.42B.300d.txt"
    embeddings = load_filtered_glove(all_features,path)
    new_model = save_preprocess_obj(embeddings,'glove.filtered')
    print "Saved"
    pass


###############################################################
def save_preprocess_obj(obj, name):
    filename = '../data/preprocess/'+name + '.pkl'
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_preprocess_obj(name):
    with open('../data/preprocess/'+name + '.pkl', 'rb') as f:
        return pickle.load(f)

###############################################################

if __name__ == '__main__':
    save_new_glove_model()
    # domain = "reddit"
    # domain = "twitter"
    # read_labeled(domain)

    