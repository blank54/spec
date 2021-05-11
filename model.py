#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
import codecs
import numpy as np
import pickle as pk
from time import time
from tqdm import tqdm

from keras import Model
from keras.optimizers import Adam
from keras_bert import Tokenizer
from keras_bert import load_trained_model_from_checkpoint
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Input
from keras.layers import Dense, Bidirectional, LSTM, TimeDistributed
from keras_contrib.layers import CRF

from config import Config
with open('/data/blank54/workspace/project/spec/spec.cfg', 'r') as f:
    cfg = Config(f)

sys.path.append(cfg['root'])
from object import NER_Dataset, NER_Result
from analysis import Utils
utils = Utils()


class Word2VecModel:
    def __init__(self, model, parameters):
        self.docs = ''
        self.model = model
        self.parameters = parameters

        self.feature_size = self.model.wv.vector_size
        self.word_vector = ''

    def __len__(self):
        return len(self.word_vector)

    def train(self, docs, update=False):
        if update:
            self.model.min_count = 0
        else:
            pass

        self.model.build_vocab(sentences=docs, update=update)
        self.model.train(sentences=docs,
                         total_examples=self.model.corpus_count,
                         epochs=self.model.epochs)
        self.word_vector = self.__get_word_vector()

    def __get_word_vector(self):
        word_vector = {w: self.model.wv[w] for w in self.model.wv.vocab.keys()}
        word_vector['__PAD__'] = np.zeros(self.feature_size)
        word_vector['__UNK__'] = np.zeros(self.feature_size)
        return word_vector


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


class NER_Model:
    def __init__(self, **kwargs):
        self.fdir = os.path.join(cfg['root'], cfg['fdir_ner_model'])
        self.fname = kwargs.get('fname', '')
        self.fpath = os.path.join(self.fdir, self.fname)

        self.max_sent_len = ''
        self.feature_size = ''
        self.embedding_model = ''

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

        self.confusion_matrix = ''
        self.f1_score_list = ''
        self.f1_score_average = ''

    def initialize(self, **kwargs):
        if 'ner_corpus' in kwargs.keys():
            ner_corpus = kwargs.get('ner_corpus', '')
            self.max_sent_len = ner_corpus.max_sent_len
            self.feature_size = ner_corpus.feature_size
            self.embedding_model = ner_corpus.embedding_model
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

    def __get_confusion_matrix(self):
        matrix_size = len(self.ner_labels)-2
        matrix = np.zeros((matrix_size+1, matrix_size+1), dtype='int64')

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
        self.confusion_matrix = matrix

    def __get_f1_score(self):
        self.f1_score_list, self.f1_score_average = utils.f1_score_from_matrix(self.confusion_matrix)

    def evaluate(self):
        self.__get_confusion_matrix()
        self.__get_f1_score()

        print('|--------------------------------------------------')
        print('|Confusion Matrix:')
        print(self.confusion_matrix)
        print('|--------------------------------------------------')
        print('|F1 Score: {:.03f}'.format(self.f1_score_average))
        print('|--------------------------------------------------')
        for category, f1_score in zip(self.ner_labels, self.f1_score_list):
            print('|    [{}]: {:.03f}'.format(category, f1_score))

    def predict(self, sent):
        sent_by_id = []
        for w in [w.lower() for w in sent]:
            if w in self.word2id.keys():
                sent_by_id.append(self.word2id[w])
            else:
                sent_by_id.append(self.word2id['__UNK__'])

        sent_pad = pad_sequences(maxlen=self.max_sent_len, sequences=[sent_by_id], padding='post', value=self.word2id['__PAD__'])
        X_input = np.zeros((1, self.max_sent_len, self.feature_size), dtype=list)
        for j, w_id in enumerate(sent_pad[0]):
            for k in range(self.feature_size):
                word = self.id2word[w_id]
                X_input[0, j, k] = self.embedding_model.word_vector[word][k]

        prediction = self.model.predict(X_input)
        pred_labels = self.__pred2labels(sents=sent_pad, prediction=prediction)[0]
        return NER_Result(input_sent=sent, pred_labels=pred_labels)


class BERT_Tokenizer(Tokenizer):
    def _tokenize(self, text):
        if not self._cased:
            text = text.lower()
            
        spaced = ''
        for ch in text:
            if self._is_punctuation(ch) or self._is_cjk_character(ch):
                spaced += ' ' + ch + ' '
            elif self._is_space(ch):
                spaced += ' '
            elif ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch
                
        tokens = []
        for word in spaced.strip().split():
            tokens += self._word_piece_tokenize(word)
        return tokens


class BERT:
    def __init__(self, fdir_pretrained, SEQ_LEN):
        self.fdir_pretrained = fdir_pretrained
        self.fpath_config = os.path.join(self.fdir_pretrained, 'bert_config.json')
        self.fpath_checkpoint = os.path.join(self.fdir_pretrained, 'bert_model.ckpt')

        self.SEQ_LEN = SEQ_LEN # maximum length of input sentence

        self.LEFT_COLUMN = 'left_text'
        self.RIGHT_COLUMN = 'right_text'
        self.LABEL_COLUMN = 'label'

    def build_vocab(self):
        fpath_vocab = os.path.join(self.fdir_pretrained, 'vocab.txt')

        token_dict = {}
        with codecs.open(fpath_vocab, 'r', encoding='utf-8') as reader:
            for line in reader:
                token = line.strip()
                if '_' in token:
                    token = token.replace('_', '')
                    token = '##' + token
                token_dict[token] = len(token_dict)

        init_len = len(token_dict)

        # for doc in read.docs(iter_unit='paragraph'):
        #     for domain_token in [t.lower() for t in doc.token]:
        #         if domain_token not in token_dict.keys():
        #             token = '##' + domain_token
        #             token_dict[token] = len(token_dict)
        #         else:
        #             continue
        updated_len = len(token_dict)
        
        print('Build BERT Vocabs')
        print('    | Initial: {}'.format(init_len))
        print('    | Updated: {}'.format(updated_len))
        reverse_dict = {i: t for t, i in token_dict.items()}

        return token_dict, reverse_dict

    def data2input(self, token_dict, data, option):
        tokenizer = BERT_Tokenizer(token_dict)
        
        data[self.LEFT_COLUMN] = data[self.LEFT_COLUMN].astype(str)
        data[self.RIGHT_COLUMN] = data[self.RIGHT_COLUMN].astype(str)

        indices, targets = [], []
        for idx in tqdm(range(len(data))):
            ids, segments = tokenizer.encode(data[self.LEFT_COLUMN].iloc[idx], data[self.RIGHT_COLUMN].iloc[idx], max_len=self.SEQ_LEN)
            indices.append(ids)

            if option=='train' or option=='test':
                target = [0]*5 #
                target[data[self.LABEL_COLUMN].iloc[idx]] = 1
                targets.append(target)
            else:
                continue
            
        X = [np.array(indices), np.zeros_like(indices)]
        if option=='train' or option=='test':
            Y = np.array(targets)
            return X, Y
        elif option=='predict':
            return X
        else:
            print('ERROR: Wrong option!!!')
            return None

    def load_model(self):
        layer_num = 12
        model = load_trained_model_from_checkpoint(
            config_file=self.fpath_config,
            checkpoint_file=self.fpath_checkpoint,
            training=True,
            trainable=True,
            seq_len=self.SEQ_LEN,
            )

        return model

    def save_model(self, model, fpath):
        model.save_weights(fpath)

        fdir, fname = os.path.split(fpath)
        print('Trained BERT-based ProvisionPairing Model:')
        print('    | fdir : {}'.format(fdir))
        print('    | fname: {}'.format(fname))

    def fine_tuning(self, model):
        inputs = model.inputs[:2]
        dense = model.layers[-3].output
        outputs = Dense(
            units=5,
            activation='sigmoid',
            name='ProvisionPairLabel'
        )(dense)

        tuned_model = Model(inputs=inputs, outputs=outputs)
        tuned_model.compile(
            optimizer=Adam(learning_rate=0.00001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return tuned_model

    def train(self, data, model, parameters):
        train_X, train_Y, test_X, test_Y = data[0], data[1], data[2], data[3]
        epochs = parameters.get('epochs', 1)
        batch_size = parameters.get('batch_size', 32)
        verbose = parameters.get('verbose', 1)
        shuffle = parameters.get('shuffle', 1)

        history = model.fit(train_X, train_Y,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=verbose,
                        validation_data=(test_X, test_Y),
                        shuffle=shuffle)

        return model