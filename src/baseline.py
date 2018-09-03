# -*- coding: utf-8 -*- 
"""
Baseline methods

 Xia Cui
 08/2018
"""

import sys
import glob
import random
import pickle,os
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from operator import add
from feature import get_features

def read_labeled(domain):
    input_file = open('../data/%s.labeled'%domain,'r')
    lines = input_file.readlines()
    labels = [int(line.strip().split()[0]) for line in lines]
    contents = [line.strip().split()[1:] for line in lines]
    contents = [[word.lower() for word in line] for line in contents]
    return contents,labels


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
        all_data += read_labeled(domain)[0]
    all_features = set([x for reivew in all_data for x in reivew])
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

def make_sentence_vector(sentence,embeddings):
    temp = np.array(np.zeros(300))
    # print embeddings
    for word in sentence:
        if word in embeddings:
            temp = map(add, temp, np.array(embeddings[word]))
        else:
            # print "%s is not in pretrained embeddings"%word
            temp = map(add,temp,np.array(np.zeros(300)))
            # temp =np.add(temp,embedding_for_word(word,pre_model,self_model))
        # print len(temp)
    return temp

def set_up_data(sentences,embeddings):
    u = list()
    for sent in sentences:
        u.append(make_sentence_vector(sent,embeddings))
    return u

# expand features
def concatenate(a,b):
    if len(a)>0 and len(b)>0:
        print len(a),len(b)
        return np.concatenate((a,b),axis=1)
    elif len(a)==0 and len(b)!=0:
        print "a empty!!"
        return np.array(b)
    elif len(b)==0 and len(a)!=0: 
        print "b empty!! length a = %d"%len(a)
        return np.array(a)
    else:
        return list()
pass

###############################################################
def store_writeprints(domain):
    X,y = read_labeled(domain)
    docs = [' '.join(x) for x in X]
    X_2 = get_features(docs)
    save_preprocess_obj(dict(zip(X,X_2)),'%s.writeprints'%domain)
    pass

def prepare_data(domain,embeddings,k=None,selected_labels=None,writeprints=False):
    X,y = read_labeled(domain)
    if k !=None:
        # selected_labels = random.sample(set(y),k)
        X,y = filter_labels(X,y,selected_labels)
    if writeprints == False:
        X = set_up_data(X,embeddings)
    else:
        docs = [' '.join(x) for x in X]
        add_X = get_features(docs)
        X = set_up_data(X,embeddings)
        X = concatenate(X,add_X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print len(X_train),len(X_test)
    if k == None:
        filename = "../data/%s/X_train"%domain
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
        np.save("../data/%s/X_train"%domain,X_train)
        np.save("../data/%s/X_test"%domain,X_test)
        np.save("../data/%s/y_train"%domain,y_train)
        np.save("../data/%s/y_test"%domain,y_test)
    else:
        filename = "../data/%s/%s/X_train"%(domain,k)
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
        np.save("../data/%s/%s/X_train"%(domain,k),X_train)
        np.save("../data/%s/%s/X_test"%(domain,k),X_test)
        np.save("../data/%s/%s/y_train"%(domain,k),y_train)
        np.save("../data/%s/%s/y_test"%(domain,k),y_test)
    return X_train,y_train,X_test,y_test

def evaluate_pair(source,target,k=None):
    print source,target
    print k
    # initialize
    X_train,y_train,X_test,y_test = None,None,None,None
    if k == None:
        X_train = np.load("../data/%s/X_train.npy"%source)
        X_test = np.load("../data/%s/X_test.npy"%target)
        y_train = np.load("../data/%s/y_train.npy"%source)
        y_test = np.load("../data/%s/y_test.npy"%target)
    else:
        X_train = np.load("../data/%s/%s/X_train.npy"%(source,k))
        X_test = np.load("../data/%s/%s/X_test.npy"%(target,k))
        y_train = np.load("../data/%s/%s/y_train.npy"%(source,k))
        y_test = np.load("../data/%s/%s/y_test.npy"%(target,k))
    print 'glove+writeprints:',baseline(X_train,y_train,X_test,y_test)
    print 'glove:',baseline(X_train[:300],y_train,X_test[:300],y_test)
    pass

def filter_labels(X,y,selected_labels):
    # print len(X),len(y)
    # print selected_labels
    new_X = [a for a,b in zip(X,y) if b in selected_labels]
    new_y = [b for b in y if b in selected_labels]
    # print len(X),len(new_X)
    return new_X,new_y

# baseline: NoAdapt
# default: LogisticRegression
def baseline(X_train,y_train,X_test,y_test,clf='lr'):
    clf_func = get_clf_func(clf)
    clf_func.fit(X_train,y_train)
    pred = clf_func.predict(X_test)
    acc = accuracy_score(y_test, pred)
    # print acc
    return acc

# stroe all the classifiers and get the right one to be used
def get_clf_func(clf,k=15):
    if clf == 'knn':
        clf_func = KNeighborsClassifier(n_neighbors=k) # knn
    elif clf == 'lr':
        clf_func = LogisticRegression(n_jobs=-1,solver='lbfgs')
    elif clf == 'tree':
        clf_func = DecisionTreeClassifier()
    elif clf == 'naive':
        clf_func = GaussianNB()
    elif clf == 'svm':
        clf_func = LinearSVC(random_state=0)
    else: # nn
        clf_func = MLPClassifier()
    return clf_func

def preprocess(k):
    # save_new_glove_model() 
    # authors = load_preprocess_obj('reddit_author_dict').values()
    # selected_labels = random.sample(authors,k)
    # np.save('../data/authors_%s'%k,selected_labels)
    selected_labels = np.load('../data/authors_%s.npy'%k)
    print selected_labels
    embeddings = load_preprocess_obj('glove.filtered')
    domains = ['reddit','twitter']
    for domain in domains:
        prepare_data(domain,embeddings,k,selected_labels,writeprints=True)
    pass

###############################################################

if __name__ == '__main__':
    k = 10
    # k = 25
    # k = 50
    # preprocess(k)

    domain = "reddit"
    # domain = "twitter"
    X,y = read_labeled(domain)
    X = X[:10]
    print X
    
    # store_writeprints(domain)
    # source = "reddit"
    # target = "twitter"
    # source = "twitter"
    # target = "reddit"
    # evaluate_pair(source,target,k)