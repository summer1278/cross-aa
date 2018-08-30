# -*- coding: utf-8 -*- 
"""
Feature set

 Xia Cui
 08/2018
"""
import re
import string
import nltk
import numpy as np
from nltk.data import load

def writeprints(text):
    writeprints = []
    """word-level"""
    # total words
    words = re.split("\W+",text.lower())
    num_words = len(filter(None,words))
    # number of words
    # print num_words
    word_lengths = np.array([len(word) for word in words])
    # average word length
    avg_word_length = word_lengths.mean()
    # word length ditribution
    std_word_length = word_lengths.std()
    # vocab richness
    vocab_richness = float(len(set(words)))/len(words)
    # print vocab_richness
    """char-level"""
    tokens = nltk.word_tokenize(text)
    # number of chars
    num_chars = sum([len(w) for w in tokens])
    # count of letters
    num_letters = len(re.findall(r'(\w)',' '.join(tokens)))
    # count of special chars
    num_specials = num_chars - num_letters
    # print num_specials
    """digit"""
    num_digits = sum([1 for s in tokens if s.isdigit()])
    # print num_digits
    
    """function words"""
    stp = nltk.corpus.stopwords.words('english')
    filtered_text = [w for w in words if not w in stp]
    ftable=nltk.FreqDist(filtered_text)
    # number of function words
    num_functs = len(ftable.keys())
    # print "function",num_functs
    """punctuation"""
    num_puncts = sum([1 for c in tokens if c in string.punctuation])
    # print num_puncts

    """POS tags"""
    tagdict = load('help/tagsets/upenn_tagset.pickle')
    tagset = tagdict.keys()
    freq_pos_tags = [sum([1 for x in token_to_pos(tokens) if x==pos]) 
        for pos in tagset]

    """misspelled words"""
    num_misspelled = sum([1 for word in words if word not in nltk.corpus.words.words()])
    # print num_misspelled
    writeprints = [num_words,avg_word_length,std_word_length,vocab_richness,num_chars,num_letters,
        num_specials,num_digits,num_functs,num_puncts,num_misspelled]+freq_pos_tags
    return writeprints

def get_features(contents):
    features = []
    for text in contents:
        features.append(writeprints(text))
    return features

def token_to_pos(tokens):
    return [p[1] for p in nltk.pos_tag(tokens)]

if __name__ == '__main__':
    # domain = "reddit"
    # read_labeled(domain)

    lines = ["109 Sometimes I have to wonder if he 's doing this out of his own accord or is it Konami forcing his studio to be a Metal Gear studio exclusively . As far as I 'm aware , Kojima wants to make something that is not Metal Gear for the longest of time ... I , for one , know Kojima 's mind is capable of much more than that ... As a group of Kotaku commenters once said , `` Kojima-san does n't have to make Metal Gear anymore . ''",
        "1 I do n't frequent that subreddit , so I would n't know .\n",
        "2 [ TIL what brogue is ] . Quite an adjective , I must say .",
        "3 & gt ; Truly August is the best 1 month Indeed .]\n"]
    docs = [' '.join(line.strip().split()[1:]) for line in lines]
    # print contents
    print len(writeprints(docs[0]))