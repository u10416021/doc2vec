# coding:utf-8
import argparse
import os
import sys
import gensim
import sklearn
import glob
import numpy as np
from gensim.models.doc2vec import Doc2Vec, LabeledSentence

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="path to input dataset")
args = vars(ap.parse_args())
f = glob.glob(str(args["dataset"])+"/"+"*.txt")

TaggededDocument = gensim.models.doc2vec.TaggedDocument

def get_datasest():
    total_words=0
    x_train = []
    for (i, txtFile) in enumerate(f):
        with open(str(txtFile),'r') as cf:
            docs = cf.readlines()
            #print(len(docs))
            total_words= len(docs)+total_words
        #y = np.concatenate(np.ones(len(docs)))
        for i, text in enumerate(docs):
            word_list = text.split(' ')
            l = len(word_list)
            word_list[l-1] = word_list[l-1].strip()

            document = TaggededDocument(word_list, tags=[i])
            x_train.append(document)

    return x_train,total_words

def test(test_text):
    model_dm = Doc2Vec.load("model_dm")
    inferred_vector_dm = model_dm.infer_vector(test_text)
    #print(inferred_vector_dm)
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)

    return sims

if __name__ == '__main__':
    (x_train,total_words) = get_datasest()
    print(total_words)
    test_text = ['fucking', 'aweful', 'shit']
    sims = test(test_text)
    for count, sim in sims:
        sentence = x_train[count]
        words = ''
        for word in sentence[0]:
            words = words + word + ' '
        print(words, sim, len(sentence[0]))
