# -*- coding: utf-8 -*- 
"""
Baseline methods for guardian dataset

 Xia Cui
 08/2018
"""
from baseline import save_preprocess_obj,concatenate,set_up_data
from feature import get_features
from adf import finder

def read_labeled(source,target,dataset):
    input_file = open('../data/train-test/%s-%s/%s'%(source,target,dataset),'r')
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
    domains = ['Politics','Society','UK','World']
    all_data = list()
    for source in domains:
        for target in domains:
            if source == target:
                continue
            all_data += read_labeled(source,target,'train')[0]
            all_data += read_labeled(source,target,'test')[0]
    all_features = set([x for reivew in all_data for x in reivew])
    print len(all_features)
    return all_features

def save_new_glove_model():
    all_features = collect_all()
    path = "../../pivot-selection/data/glove.42B.300d.txt"
    embeddings = load_filtered_glove(all_features,path)
    new_model = save_preprocess_obj(embeddings,'glove.filtered.guardian')
    print "Saved"
    pass

###############################################################

def prepare_data(source,target,embeddings,writeprints=True):
    X_train,y_train = read_labeled(source,target,'train')
    X_test,y_test = read_labeled(source,target,'test')
    if writeprints == False:
        X_train = set_up_data(X_train,embeddings)
        X_test = set_up_data(X_test,embeddings)
    else:
        docs_train = [' '.join(x) for x in X_train]
        docs_test = [' '.join(x) for x in X_test]
        add_train = get_features(docs_train)
        add_test = get_features(docs_test)
        X_train = set_up_data(X_train,embeddings)
        X_test = set_up_data(X_test,embeddings)
        X_train = concatenate(X_train,add_train)
        X_test = concatenate(X_test,add_test)

    filename = "../data/%s-%s/X_train"%(source,target)
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    np.save("../data/train-test/%s-%s/X_train"%(source,target),X_train)
    np.save("../data/%s-%s/X_test"%(source,target),X_test)
    np.save("../data/%s-%s/y_train"%(source,target),y_train)
    np.save("../data/%s-%s/y_test"%%(source,target),y_test)
    return X_train,y_train,X_test,y_test

def evaluate_pair(source,target):
    print source,target
    X_train = np.load("../data/train-test/%s-%s/X_train.npy"%(source,target))
    X_test = np.load("../data/train-test/%s-%s/X_test.npy"%(source,target))
    y_train = np.load("../data/train-test/%s-%s/y_train.npy"%(source,target))
    y_test = np.load("../data/train-test/%s-%s/y_test.npy"%(source,target))
    print 'glove:',finder(X_train[:,:300],y_train,X_test[:,:300],y_test)
    print 'glove+writeprints:',finder(X_train,y_train,X_test,y_test)
    pass

def preprocess():
    # save_new_glove_model()
    embeddings = load_preprocess_obj('glove.filtered.guardian')
    domains = ['Politics','Society','UK','World']
    for source in domains:
        for target in domains:
            if source == target:
                continue
            prepare_data(source,target,embeddings)
    pass

###############################################################

if __name__ == '__main__':
    preprocess()