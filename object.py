#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import numpy as np
import pickle as pk
from time import time

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import Dense, Bidirectional, LSTM, TimeDistributed
from keras_contrib.layers import CRF

from sklearn.model_selection import train_test_split

from config import Config
with open('/data/blank54/workspace/project/spec/spec.cfg', 'r') as f:
    cfg = Config(f)

class Doc:
    fpath = ''

    tag = ''
    text = ''

    token = ''
    stop = ''
    stem = ''
    lemma = ''
    ngram = ''


class Spec(Doc):
    def __init__(self, tag, text, **kwargs):
        self.tag = tag
        self.text = text
        self.sents = []

    def __str__(self):
        return self.text

    def __len__(self):
        return len(self.sents)


class Section(Doc):
    def __init__(self, tag, text, **kwargs):
        self.tag = tag
        self.text = text
        self.sents = []

    def __str__(self):
        return self.text

    def __len__(self):
        return len(self.sents)


class Paragraph(Doc):
    def __init__(self, tag, text, **kwargs):
        self.tag = tag
        self.text = text

    def __str__(self):
        return self.text

    def __len__(self):
        return len(self.text)


class Sentence(Doc):
    def __init__(self, tag, text, **kwargs):
        self.tag = tag
        self.text = text
        self.pos = []

    def __str__(self):
        return self.text

    def __len__(self):
        return len(self.text)


class WordFlow:
    def __init__(self, word_from, word_to, similarity, score):
        self.word_from = word_from
        self.word_to = word_to
        self.similarity = similarity
        self.score = score

    def __str__(self):
        return '{} -> {} ({:.03f})'.format(self.word_from, self.word_to, self.score)


class Word2VecModel:
    def __init__(self, docs, model, parameters):
        self.docs = docs
        self.model = model
        self.parameters = parameters

    def train(self, **kwargs):
        self.model.build_vocab(sentences=self.docs)
        self.model.train(sentences=self.docs,
                         total_examples=self.model.corpus_count,
                         epochs=self.model.epochs)

    def update(self, new_docs, min_count=0):
        self.model.min_count = min_count
        self.model.build_vocab(sentences=new_docs, update=True)
        self.model.train(sentences=new_docs, total_examples=self.model.corpus_count, epochs=self.model.iter)


class Doc2VecModel:
    def __init__(self, docs, model, parameters):
        self.docs = docs
        self.model = model
        self.parameters = parameters

    def train(self, **kwargs):
        self.model.build_vocab(documents=self.docs)
        self.model.train(documents=self.docs,
                         total_examples=self.model.corpus_count,
                         epochs=self.model.epochs)


# class Pair:
#     def __init__(self, tags, score):
#         self.tags = tags
#         self.score = score


class LabeledSentence:
    def __init__(self, tag, words, labels):
        self.tag = tag
        self.words = words
        self.labels = labels

    def __len__(self):
        return len(self.words)

    def __str__(self):
        return ' '.join(self.words)


class NER_Labels:
    def __init__(self, label_list):
        self.label_list = label_list

        self.label2id = {str(l): int(i) for l, i in self.label_list}
        self.id2label = {int(i): str(l) for l, i in self.label_list}

    def __call__(self):
        return self.label_list

    def __len__(self):
        return len(self.label_list)


class NER_Corpus:
    def __init__(self, fpath, docs, ner_labels, weighted_labels, word2id, id2word, 
        max_sent_len, X_words, Y_labels, X_words_pad, Y_labels_pad):
        self.fpath = fpath

        self.docs = docs
        self.ner_labels = ner_labels
        self.weighted_labels = weighted_labels

        self.word2id = word2id
        self.id2word = id2word

        self.max_sent_len = max_sent_len
        self.feature_size = ''

        self.X_words = X_words
        self.Y_labels = Y_labels
        self.X_words_pad = X_words_pad
        self.Y_labels_pad = Y_labels_pad

        self.X_embedded = ''
        self.Y_embedded = ''

    def __len__(self):
        return len(self.X_words)


class NER_Dataset:
    def __init__(self, X, Y, test_size):
        self.test_size = test_size
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=self.test_size)


class NER_Model:
    def __init__(self, **kwargs):
        self.fdir = os.path.join(cfg['root'], cfg['fdir_model'], 'ner/')
        self.fname = kwargs.get('fname', '')
        self.fpath = os.path.join(self.fdir, self.fname)

        self.max_sent_len = ''
        self.feature_size = ''
        self.ner_labels = {}
        self.word2id = {}
        self.id2word = {}

        self.parameters = ''
        self.lstm_units = ''
        self.lstm_return_sequences = ''
        self.lstm_recurrent_dropout = ''
        self.dense_units = ''
        self.dense_activation = ''
        self.test_size = ''

        self.dataset = ''
        self.model = ''

    def initialize(self, **kwargs):
        if 'ner_corpus' in kwargs.keys():
            ner_corpus = kwargs.get('ner_corpus', '')
            self.max_sent_len = ner_corpus.max_sent_len
            self.feature_size = ner_corpus.feature_size
            self.ner_labels = ner_corpus.ner_labels
            self.word2id = ner_corpus.word2id
            self.id2word = ner_corpus.id2word
            del ner_corpus
        else:
            pass

        if 'parameters' in kwargs.keys():
            parameters = kwargs.get('parameters')
            self.lstm_units = parameters.get('lstm_units')
            self.lstm_return_sequences = parameters.get('lstm_return_sequences')
            self.lstm_recurrent_dropout = parameters.get('lstm_recurrent_dropout')
            self.dense_units = parameters.get('dense_units')
            self.dense_activation = parameters.get('dense_activation')
        else:
            pass

        _input = Input(shape=(self.max_sent_len, self.feature_size))
        model = Bidirectional(LSTM(units=self.lstm_units,
                                   return_sequences=self.lstm_return_sequences,
                                   recurrent_dropout=self.lstm_recurrent_dropout))(_input)
        model = TimeDistributed(Dense(units=self.dense_units,
                                      activation=self.dense_activation))(model)
        crf = CRF(len(self.ner_labels))
        _output = crf(model)

        model = Model(inputs=_input, outputs=_output)
        model.compile(optimizer='rmsprop',
                      loss=crf.loss_function,
                      metrics=[crf.accuracy])
        
        self.model = model
        print('NER Model Initialization')

    def train(self, X, Y, **kwargs):
        if 'parameters' in kwargs.keys():
            parameters = kwargs.get('parameters')
            self.test_size = parameters.get('test_size')
            self.batch_size = parameters.get('batch_size')
            self.epochs = parameters.get('epochs')
            self.validation_split = parameters.get('validation_split')
        else:
            pass

        _start = time()
        self.dataset = NER_Dataset(X=X, Y=Y, test_size=self.test_size)
        self.model.fit(x=self.dataset.X_train,
                       y=self.dataset.Y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       validation_split=self.validation_split,
                       verbose=True)
        _end = time()
        print('NER Model Training: {:,} records in {:,.02f} minutes'.format(len(self.dataset.X_train), (_end-_start)/60))

    def save(self, **kwargs):
        fdir = kwargs.get('fdir', self.fdir)
        fname = kwargs.get('fname', self.fname)
        fpath = os.path.join(fdir, fname)
        fpath_dataset = os.path.join(fdir, '{}_dataset.pk'.format(fname[:-3]))
        
        os.makedirs(fdir, exist_ok=True)
        self.model.save(fpath)

        with open(fpath_dataset, 'wb') as f:
            pk.dump(self.dataset, f)

        print('Save NER Model: {}'.format(fpath))

    def load(self, ner_corpus, parameters, **kwargs):
        self.initialize(ner_corpus=ner_corpus, parameters=parameters)
        self.model.load_weights(self.fpath)

        fdir = kwargs.get('fdir', self.fdir)
        fpath_dataset = os.path.join(fdir, '{}_dataset.pk'.format(self.fname[:-3]))
        with open(fpath_dataset, 'rb') as f:
            self.dataset = pk.load(f)

    def __pred2labels(self, sents, prediction):
        pred_labels = []
        for sent, pred in zip(sents, prediction):
            try:
                sent_len = np.where(sent==self.word2id['__PAD__'])[0][0]
            except:
                sent_len = self.max_sent_len
                
            labels = []
            for i in range(sent_len):
                labels.append(self.ner_labels.id2label[np.argmax(pred[i])])
            pred_labels.append(labels)
        return pred_labels

    def evaluate(self):
        matrix_size = len(self.ner_labels)-2
        matrix = np.zeros((matrix_size+1, matrix_size+1))

        prediction = self.model.predict(self.dataset.X_test)
        pred_labels = self.__pred2labels(self.dataset.X_test, prediction)
        test_labels = self.__pred2labels(self.dataset.Y_test, self.dataset.Y_test)

        for i in range(len(pred_labels)):
            for j, pred in enumerate(pred_labels[i]):
                matrix[self.ner_labels.label2id[test_labels[i][j]], self.ner_labels.label2id[pred]] += 1
                
        for i in range(matrix_size):
            matrix[i, matrix_size] = sum(matrix[i, 0:matrix_size])
            matrix[matrix_size, i] = sum(matrix[0:matrix_size, i])
            
        matrix[matrix_size, matrix_size] = sum(matrix[matrix_size, 0:matrix_size])
        print(matrix)
        return matrix

    # def predict(self, new_sent):
    #     '''
    #     new_sent: list of lowercase words (e.g., [w1, w2, ...])
    #     '''
    #     sent_by_id = []
    #     for w in [w.lower() for w in sent]:
    #         try:
    #             sent_by_id.append(self.word2id[w])