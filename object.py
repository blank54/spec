#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os


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