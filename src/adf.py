"""
Augmented Doppelganger Finder Implementation
(Overdorf and Greenstadt, 2016)

 Xia Cui
 09/2018
"""
from itertools import combinations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def classify(X_train,y_train,X_test,y_test):
    clf_func = LogisticRegression(n_jobs=-1,solver='lbfgs')
    clf_func.fit(X_train,y_train)
    labels_proba = clf_func.predict_proba(X_test)
    clf_classes = clf_func.classes_
    average_proba = np.average(labels_proba,axis=0)
    return average_proba,clf_classes

def mixed_training(X_train,y_train):
    proba_dict = dict()
    classes = set(y_train)
    for target_class in classes:
        new_X_train,new_y_train,new_X_test,new_y_test = new_train(X_train,y_train,target_class)
        average_proba,clf_classes =classify(new_X_train,new_y_train,new_X_test,new_y_test)
        for x,y in zip(average_proba,clf_classes):
            proba_dict[(target_class,y)]=x 
    # print proba_dict
    return proba_dict

def ensemble(proba_dict,s_classes,t_classes):
    # pairs = combinations(classes,2)
    pair_proba = dict()
    # print pairs
    for a in s_classes:
        temp = 0
        best = ''
        for b in t_classes:
            p = proba_dict[(a,b)]*proba_dict[(b,a)]
            if p>temp:
                temp = p
                best = b
        pair_proba[(a,best)] = temp
    print pair_proba
    return pair_proba.keys()

def new_train(X_train,y_train,target_class):
    # print target_class
    new_X_train= [x for (x,y) in zip(X_train,y_train) if y != target_class]
    new_y_train = [x for x in y_train if x != target_class]
    new_X_test= [x for (x,y) in zip(X_train,y_train) if y == target_class]
    new_y_test = [x for x in y_train if x == target_class]
    # print new_X_train,new_y_train,new_X_test,new_y_test
    return new_X_train,new_y_train,new_X_test,new_y_test

def domain_label(domain,y):
    y = [domain[0]+"_"+str(x) for x in y]
    return y

def get_acc(tuples):
    a = [int(x.replace('s_','')) for (x,y) in tuples]
    b = [int(y.replace('t_','')) for (x,y) in tuples]
    acc = accuracy_score(a,b)
    return acc

def concatenate(a,b):
    if len(a)>0 and len(b)>0:
        # print len(a),len(b)
        return np.concatenate((a,b),axis=0)
    elif len(a)==0 and len(b)!=0:
        print "a empty!!"
        return np.array(b)
    elif len(b)==0 and len(a)!=0: 
        print "b empty!! length a = %d"%len(a)
        return np.array(a)
    else:
        return list()
pass

def finder(X_train,y_train,X_test,y_test):
    X_train = concatenate(X_train,X_test)
    s_classes,t_classes = domain_label('s',y_train),domain_label('t',y_test)
    y_train = s_classes + t_classes
    proba_dict= mixed_training(X_train,y_train)
    return get_acc(ensemble(proba_dict,set(s_classes),set(t_classes)))

if __name__ == '__main__':
    X_train = [[1,2,4],[2,3,4],[3,2,1],[3,4,2],[2,3,2]]
    y_train = [1,2,1,3,4]
    X_test = [[1,1,1],[1,1,2],[1,5,1],[1,8,2],[2,5,1]]
    y_test = [2,1,3,2,4]
   