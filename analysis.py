#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import re
import sys
import nltk
import pickle as pk
from tqdm import tqdm
from collections import defaultdict

from config import Config
with open('/data/blank54/workspace/project/spec/spec.cfg', 'r') as f:
    cfg = Config(f)

sys.path.append(cfg['root'])
from object import *


class Utils:
    def fname2spec_info(self, fname):
        return '_'.join(fname.split('_')[:3])

    def line2paragraph_info(self, line):
        paragraph_info = []
        for idx, item in enumerate(line.strip().split('  ')):
            if idx < 3:
                paragraph_info.append(item)
            else:
                paragraph_info.append(item.split()[0])
        return '_'.join(paragraph_info)


class BuildCorpus:
    def section2sentence(self, fpath_section):
        # def __get_starting_tag():
        #     fname_section = os.path.basename(fpath_section)
        #     spec_info = Utils().fname2spec_info(fname_section)
        #     starting_tag = '  '.join(spec_info)
        #     return starting_tag

        with open(fpath_section, 'r', encoding='utf-8') as f:
            section = re.sub('\n+\n', '\n\n', f.read().replace('\ufeff', ''))

        sentences = []
        for paragraph in section.split('\n\n'):
            lines = paragraph.split('\n')
            paragraph_tag = Utils().line2paragraph_info(lines[0])
            paragraph_sentences = lines[1:]
            for idx, sentence in enumerate(paragraph_sentences):
                tag = '{}_{:02d}'.format(paragraph_tag, idx+1)
                sentences.append(Sentence(tag=tag, text=sentence))

        return sentences


class Corpus:
    def __init__(self):
        self.fdir = os.path.join(cfg['root'], cfg['fdir_corpus'])

    def load_single(self, fname):
        with open(os.path.join(self.fdir, fname), 'rb') as f:
            return pk.load(f)

    def load_multiple(self, fdir):
        flist = os.listdir(fdir)
        for fname in flist:
            fpath = os.path.join(fdir, fname)
            with open(fpath, 'rb') as f:
                yield pk.load(f)

    def load_stopword_list(self):
        fpath_stopword_list = os.path.join(cfg['root'], cfg['fdir_corpus'], 'thesaurus/stopword_list.txt')
        with open(fpath_stopword_list, 'r', encoding='utf-8') as f:
            stopword_list = [w.strip() for w in f.read().strip().split('\n')]
        return stopword_list

class Preprocessor():
    def __remove_trash_characters(self, text):
        text = re.sub('[^ \'\?\./0-9a-zA-Zㄱ-힣\n]', '', text)

        text = text.replace(' / ', '/')
        text = re.sub('\.+\.', ' ', text)
        text = text.replace('\\\\', '\\').replace('\\r\\n', '')

        text = text.replace('\n', '  ')
        text = re.sub('\. ', '  ', text)
        text = re.sub('\s+\s', ' ', text).strip()
        return text

    def text_cleaning(self, fpath, do, **kwargs):
        if do:
            sents = kwargs.get('sents')
            sents_cleaned = []
            for s in sents:
                cleaned_text = self.__remove_trash_characters(s.text)
                sents_cleaned.append(Sentence(tag=s.tag, text=cleaned_text.lower()))

            with open(fpath, 'wb') as f:
                pk.dump(sents_cleaned, f)

        else:
            with open(fpath, 'rb') as f:
                sents_cleaned = pk.load(f)        

        return sents_cleaned

    def normalize(self, fpath, do, **kwargs):
        def __read_unit_list():
            fpath_unit_list = '/data/blank54/workspace/project/spec/corpus/thesaurus/unit_list.txt'
            unit_map = defaultdict(str)
            with open(fpath_unit_list, 'r', encoding='utf-8') as f:
                for pair in f.read().strip().split('\n'):
                    l, r = pair.split('  ')
                    unit_map[l] = r
            return unit_map

        def __normalize_unit(text, unit_map):
            words = text.split(' ')
            words_unit_normalized = []
            for w in words:
                if w in unit_map.keys():
                    words_unit_normalized.append(unit_map[w])
                else:
                    words_unit_normalized.append(w)
            return ' '.join(words_unit_normalized)

        if do:
            normalized_tokens = defaultdict(list)
            sents_normalized = []
            
            sents = kwargs.get('sents')
            re_num = re.compile('\d+\.*\d*\.*\d*')
            re_url = re.compile('.*www.*')

            with tqdm(total=len(sents)) as pbar:
                for s in sents:
                    tag, text = s.tag, s.text
                    if re_num.findall(text):
                        text = re_num.sub('NUM', text)
                        normalized_tokens['NUM'].append(tag)

                    if re_url.findall(text):
                        text =re_url.sub('URL', text)
                        normalized_tokens['URL'].append(tag)

                    unit_map = __read_unit_list()
                    if any([w for w in text.split(' ') if w in unit_map]):
                        text = __normalize_unit(text, unit_map)
                        normalized_tokens['UNIT'].append(tag)

                    sents_normalized.append(Sentence(tag=tag, text=text))
                    pbar.update(1)

            with open(fpath, 'wb') as f:
                pk.dump((sents_normalized, normalized_tokens), f)

        else:
            with open(fpath, 'rb') as f:
                sents_normalized, normalized_tokens = pk.load(f)            

        return sents_normalized, normalized_tokens

    def extract_sent_verb(self, fpath, do, **kwargs):
        if do:
            sents = kwargs.get('sents')
            verb_tag_list = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
            sents_verb = []
            with tqdm(total=len(sents)) as pbar:
                for s in sents:
                    pos_tags = nltk.pos_tag(s.text.split())
                    if any([t for w, t in pos_tags if t in verb_tag_list]):
                        s.pos = pos_tags
                        sents_verb.append(s)
                    else:
                        pass
                    pbar.update(1)

            with open(fpath, 'wb') as f:
                pk.dump(sents_verb, f)

        else:
            with open(fpath, 'rb') as f:
                sents_verb = pk.load(f)

        return sents_verb

    def stopword_removal(self, fpath, do, **kwargs):
        if do:
            sents = kwargs.get('sents')
            sents_wo_stop = []
            stopword_list = Corpus().load_stopword_list()

            with tqdm(total=len(sents)) as pbar:
                for s in sents:
                    s.text = ' '.join([w for w in s.text.split() if w not in stopword_list])
                    s.pos = [(w, t) for (w, t) in s.pos if w not in stopword_list]
                    sents_wo_stop.append(s)
                    pbar.update(1)

            with open(fpath, 'wb') as f:
                pk.dump(sents_wo_stop, f)

        else:
            with open(fpath, 'rb') as f:
                sents_wo_stop = pk.load(f)

        return sents_wo_stop

    def chunking(self, fpath, do, **kwargs):
        if do:
            sents = kwargs.get('sents')
            sents_chunk = []
            grammar = '''
                NP: {<DT>?<RB>?<JJ>*<NN.*>+}
                VP: {<M.*>?<V.*>+<IN|TO>?}
            '''

            parser = nltk.RegexpParser(grammar)
            with tqdm(total=len(sents)) as pbar:
                for s in sents:
                    s.chunk = parser.parse(s.pos)
                    sents_chunk.append(s)
                    pbar.update(1)

            with open(fpath, 'wb') as f:
                pk.dump(sents_chunk, f)

        else:
            with open(fpath, 'rb') as f:
                sents_chunk = pk.load(f)

        return sents_chunk


