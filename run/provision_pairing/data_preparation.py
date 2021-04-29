#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys

from config import Config
with open('/data/blank54/workspace/project/spec/spec.cfg', 'r') as f:
    cfg = Config(f)

sys.path.append(cfg['root'])
from analysis import Read
read = Read()


docs = read.docs(iter_unit='paragraph')
for doc in docs[:5]:
    print(doc.tag)
    print(doc.text)
    print('___________________________')