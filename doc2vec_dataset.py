# coding:utf-8
import argparse
import os
import sys
import gensim
import sklearn
import glob
import numpy as np
from gensim.models.doc2vec import Doc2Vec, LabeledSentence

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="path to input dataset")
args = vars(ap.parse_args())
f = glob.glob(str(args["dataset"])+"/"+"*.txt")

TaggededDocument = gensim.models.doc2vec.TaggedDocument

def get_datasest():
    x_train = []
    for (i, txtFile) in enumerate(f):
        with open(str(txtFile),'r') as cf:
            docs = cf.readlines()
            print(len(docs))
        #y = np.concatenate(np.ones(len(docs)))
        for i, text in enumerate(docs):
            word_list = text.split(' ')
            l = len(word_list)
            word_list[l-1] = word_list[l-1].strip()

            document = TaggededDocument(word_list, tags=[i])
            x_train.append(document)

    return x_train

def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]].reshape(1, size)) for z in corpus]
    return np.concatenate(vecs)

def train(x_train, size=200, epoch_num=1):
    model_dm = Doc2Vec(x_train,min_count=1, window = 3, size = size, sample=1e-3, negative=5, workers=4)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)
    model_dm.save('model/model_dm')

    return model_dm

def test():
    model_dm = Doc2Vec.load("model/model_dm")
    test_text = ['superb', 'wonderful']
    inferred_vector_dm = model_dm.infer_vector(test_text)
    print(inferred_vector_dm)
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)


    return sims

if __name__ == '__main__':
    x_train = get_datasest()
    model_dm = train(x_train)
    #model_dm = Doc2Vec.load("model/model_dm")
    sims = test()
    for count, sim in sims:
        sentence = x_train[count]
        words = ''
        for word in sentence[0]:
            words = words + word + ' '
        print(words, sim, len(sentence[0]))
