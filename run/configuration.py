#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
import itertools

from config import Config
with open('/data/blank54/workspace/project/spec/spec.cfg', 'r') as f:
    cfg = Config(f)

sys.path.append(cfg['root'])
from analysis import Read
read = Read()


paragraphs = [p for p in read.docs(iter_unit='paragraph')]
print(len(paragraphs))

sentences = list(itertools.chain(*[p.text.split('  ') for p in paragraphs]))
print(len(sentences))