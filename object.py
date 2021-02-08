#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os


class Spec:
    def __init__(self, tag, text, **kwargs):
        self.tag = tag
        self.text = text
        self.sents = []

    def __str__(self):
        return self.text

    def __len__(self):
        return len(self.sents)


class Sentence:
    def __init__(self, tag, text, **kwargs):
        self.tag = tag
        self.text = text
        self.pos = []
        self.chunk = []

    def __str__(self):
        return self.text

    def __len__(self):
        return len(self.text)


class Word2VecModel:
    def __init__(self, docs, model, parameters):
        self.docs = docs
        self.model = model
        self.parameters = parameters

    def __len__(self):
        return len(self.docs)

    def __call__(self):
        return self.model