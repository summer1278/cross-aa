# -*- coding: utf-8 -*- 
"""
 Authorship Dataset: Guardian

 Xia Cui
 11/2017
"""
import glob
import os,io
import re
import nltk
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def read_text_from_author(source,target,opt):
    # opt: train or test
    print source,target,opt
    files = open("../aa-data/GuardianDataset1/train-test/%s-%s/labels/rlabel_%s.txt"%(source,target,opt),'r')
    labels = open("../aa-data/GuardianDataset1/train-test/%s-%s/labels/rclass_%s.txt"%(source,target,opt),'r')
    targets = [line.strip() for line in labels]
    document_names = ["../aa-data/GuardianDataset1/all_data/processedData/"+line.strip() for line in files]
    res_file = open("../aa-data/GuardianDataset1/train-test/%s-%s/labels/%s"%(source,target,opt),'w')
    for i,label in enumerate(targets):
        # print document_names[i]
        F = io.open(document_names[i],errors="ignore")
        tokens = []
        for line in F:
            if line.strip():
                tokens += nltk.word_tokenize(line.strip())
        if len(tokens)>0 and label != -1 and len(re.findall(r'(\w)',' '.join(tokens)))>5:
            res_file.write('%d %s\n'%(author_to_idx(label),' '.join(tokens)))
    res_file.close()
    pass

def author_to_idx(author):
    return all_authors().index(author) if author in all_authors() else -1

def all_authors():
    all_fnames = glob.glob('../aa-data/GuardianDataset1/all_data/processedData/*.txt')
    authors = list(set([os.path.basename(fname).replace('.txt','').split('_')[0] for fname in all_fnames]))
    # print "#authors = ",len(authors)
    # write to file
    # res_file = open("../aa-data/GuardianDataset1/authors","w")
    # for author in authors:
    #     res_file.write("%s\n"%author)
    # res_file.close()
    return authors

def all_domains():
    all_fnames = glob.glob('../aa-data/GuardianDataset1/all_data/processedData/*.txt')
    domains = list(set([os.path.basename(fname).replace('.txt','').split('_')[1] for fname in all_fnames]))
    # print "#authors = ",len(authors)
    return domains

def generate_files(domains):
    opts = ['train','test']
    for source in domains:
        for target in domains:
            if source == target:
                continue
            for opt in opts:
                read_text_from_author(source,target,opt)
    pass

if __name__ == '__main__':
    all_authors()
    # source = "politics"
    # target = "society"
    # opt = "test"
    # read_text_from_author(source,target,opt)
    # generate_files(all_domains())
