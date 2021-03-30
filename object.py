#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
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