# -*- coding: utf-8 -*- 
"""
 Authorship Dataset: Reddit-Twitter

 Xia Cui
 11/2017
"""
import glob
import csv
import os,io
import nltk
import preprocessor as twp
import re
import pickle
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def read_text_from_author(domain):
    all_fnames = glob.glob('../aa-data/%s/*/*.txt'%(convert_domain(domain)))
    res_file = open("../aa-data/%s.labeled" %domain,'w')
    author_dict = load_preprocess_obj('%s_author_dict'%domain)
    for fname in all_fnames:
        # skip empty files
        if os.path.getsize(fname)>0: 
            with io.open(fname,errors="ignore") as F:
                print fname
                author = os.path.basename(os.path.dirname(fname))
                label = int(author_dict[author]) if author in author_dict.keys() else -1
                tokens = []
                for line in F:
                    # remove empty lines and less than 5 alphebets
                    if line.strip():
                        if domain == "twitter":
                            tokens += nltk.word_tokenize(twp.clean(line.strip().lower()))
                        else:
                            # tokens += twp.tokenize(line.lower()).strip().split()
                            temp = re.sub(r'\[http.+?\]', '', line.strip())
                            temp = re.sub(r'\(http.+?\)', '', temp)
                            tokens += nltk.word_tokenize(twp.clean(temp))
                if len(tokens)>0 and label != -1 and len(re.findall(r'(\w)',' '.join(tokens)))>5:
                    res_file.write('%d %s\n'%(label,' '.join(tokens)))
    res_file.close()
    pass

def link_author(reddit_author_dict,twitter_author_dict):
    file = open("../aa-data/index.csv","r")
    data = csv.reader(file)
    count = 0
    for line in data:
        if line:
            reddit=line[0]
            twitter=line[1].replace('@','')
            reddit_author_dict[reddit]=count
            twitter_author_dict[twitter]=count
            count += 1
    save_preprocess_obj(reddit_author_dict,'reddit_author_dict')
    save_preprocess_obj(twitter_author_dict,'twitter_author_dict')
    return reddit_author_dict,twitter_author_dict

### preparation 
# save and load after preprocessing
def save_preprocess_obj(obj, name):
    filename = '../aa-data/preprocess/'+name + '.pkl'
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_preprocess_obj(name):
    with open('../aa-data/preprocess/'+name + '.pkl', 'rb') as f:
        return pickle.load(f)

def convert_domain(domain):
    return domain.capitalize() if domain == 'reddit' else domain.capitalize()+'_text'

if __name__ == '__main__':
    # reddit_author_dict = {}
    # twitter_author_dict = {}
    # link_author(reddit_author_dict,twitter_author_dict)
    # domain = "reddit"
    domain = "twitter"
    read_text_from_author(domain)