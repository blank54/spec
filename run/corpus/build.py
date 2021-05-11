#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(1) #Do not print INFO

import re
import sys
import pickle as pk
from tqdm import tqdm

from config import Config
with open('/data/blank54/workspace/project/spec/spec.cfg', 'r') as f:
    cfg = Config(f)

sys.path.append(cfg['root'])
from object import Sentence
from analysis import BuildCorpus, Read, Write, Utils
buildcorpus = BuildCorpus()
read = Read()
write = Write()
utils = Utils()


def build_corpus_section():
    fdir_data = os.path.join(cfg['root'], cfg['fdir_data_section_manual'])
    flist = os.listdir(fdir_data)

    fdir_corpus = os.path.join(cfg['root'], cfg['fdir_corpus_section_manual'])
    with tqdm(total=len(flist)) as pbar:
        for fname in flist:
            fpath = os.path.join(fdir_data, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                tag = utils.parse_fname(fpath=fpath, iter_unit='section_manual')
                section_text = re.sub('\n+\n', '\n\n', f.read().replace('\ufeff', ''))
                s = buildcorpus.section(tag=tag, section_text=section_text)

            if s.text:
                fname_corpus = '{}.pk'.format(s.tag)
                fpath_corpus = os.path.join(fdir_corpus, fname_corpus)
                s.fpath = fpath_corpus
                write.object(obj=s, fpath=fpath_corpus)
            else:
                continue
            pbar.update(1)
    
    print('Build Corpus from ManualSection: {:,}\n └ {}'.format(len(flist), fdir_corpus))


def build_corpus_section2paragraph():
    fdir_data = os.path.join(cfg['root'], cfg['fdir_data_section_manual'])
    flist = os.listdir(fdir_data)

    fdir_corpus = os.path.join(cfg['root'], cfg['fdir_corpus_paragraph'])
    with tqdm(total=len(flist)) as pbar:
        for fname_data in flist:
            fpath_data = os.path.join(fdir_data, fname_data)
            with open(fpath_data, 'r', encoding='utf-8') as f:
                section_text = re.sub('\n+\n', '\n\n', f.read().replace('\ufeff', ''))
                for p in buildcorpus.section2paragraph(section_text=section_text):
                    if p.text:
                        fname_corpus = '{}.pk'.format(p.tag)
                        fpath_corpus = os.path.join(fdir_corpus, fname_corpus)
                        p.fpath = fpath_corpus
                        write.object(obj=p, fpath=fpath_corpus)
                    else:
                        continue
            pbar.update(1)

    print('Build Corpus (Paragraph): {:,}\n └ {}'.format(len(os.listdir(fdir_corpus)), fdir_corpus))

def build_corpus_sentence():
    sections = read.docs(iter_unit='section_manual')
    for section in sections:
        sents = section.text.split('  ')
        for idx, text in enumerate(sents):
            tag = '{}_{:03d}'.format(section.tag, idx).lower()
            fname = '{}.pk'.format(tag)
            fdir = os.path.join(cfg['root'], cfg['fdir_corpus_sentence'])
            fpath = os.path.join(fdir, fname)

            sent = Sentence(tag=tag, text=text)
            sent.fpath = fpath
            with open(sent.fpath, 'wb') as f:
                pk.dump(sent, f)

    print('Build Corpus (Sentence): {:,}\n └ {}'.format(len(os.listdir(fdir)), fdir))


if __name__ == '__main__':
    # build_corpus_section()
    # build_corpus_section2paragraph()
    build_corpus_sentence()