#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os


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