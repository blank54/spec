#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import re
import sys
import copy
import nltk
import operator
import itertools
import pickle as pk
from tqdm import tqdm
from numba import jit, cuda
from collections import defaultdict
from inflection import singularize

from gensim.models import Word2Vec

from config import Config
with open('/data/blank54/workspace/project/spec/spec.cfg', 'r') as f:
    cfg = Config(f)

sys.path.append(cfg['root'])
from object import *


class Utils:
    def fname2spec_info(self, fname):
        return '_'.join(fname[:-4].split('_')[:3])

    def line2paragraph_info(self, line):
        paragraph_info = []
        for idx, item in enumerate(line.strip().split('  ')):
            if idx < 3:
                paragraph_info.append(item)
            else:
                paragraph_info.append(item.split()[0])
        return '_'.join(paragraph_info)

    def parameters2fname(self, parameters):
        return '_'.join([str(v) for (p, v) in parameters.items()])

    def merge_chunks(self, chunks):
        result = []
        for c in chunks:
            if type(c) == nltk.tree.Tree:
                result.append('+'.join(['/'.join((w, t)) for w, t in c]))
            elif type(c) == tuple:
                result.append('/'.join(c))
            else:
                print('ERROR: Wrong Chunk Type: {}'.format(c))
        return result

    def most_probable_pos(self, word, **kwargs):
        if 'corpus' in kwargs.keys():
            corpus = kwargs.get('corpus')
        else:
            fdir_corpus = os.path.join(cfg['root'], cfg['fdir_corpus'])
            fpath_corpus = os.path.join(fdir_corpus, 'sentence', 'sentences_chunk.pk')
            corpus = Corpus().load_single(fpath=fpath_corpus)

        pos_stat = Stat().pos_count(corpus)
        tag, freq = max(pos_stat[word].items(), key=operator.itemgetter(1))
        total = sum(pos_stat[word].values())
        return (tag, '{:.02f}'.format(freq/total))



class Stat:
    def pos_count(self, sents):
        pos = defaultdict(dict)
        for s in sents:
            for (word, tag) in s.pos:
                if tag not in pos[word].keys():
                    pos[word][tag] = 1
                else:
                    pos[word][tag] += 1
        return pos

    def singularized(self, sents):
        singularized = []
        words_set = list(set(itertools.chain(*[[w for w in s.text.split()] for s in sents])))
        for w in words_set:
            if w != singularize(w):
                singularized.append('{}==>{}'.format(w, singularize(w)))

        cnt_before = len(words_set)
        cnt_after = len(singularized)
        return singularized, cnt_before, cnt_after


class BuildCorpus:
    def section2sentence(self, data):
        sentences = []
        for paragraph in data.split('\n\n'):
            lines = paragraph.split('\n')
            paragraph_tag = Utils().line2paragraph_info(lines[0])
            paragraph_sentences = lines[1:]
            for idx, sentence in enumerate(paragraph_sentences):
                tag = '{}_{:02d}'.format(paragraph_tag, idx+1)
                sentences.append(Sentence(tag=tag, text=sentence))

        return sentences

    def spec2sentence(self, info, data, min_sent_len):
        parsed = [s.strip().lower() for s in data.split('\n') if len(s.split(' ')) > min_sent_len]
        
        fully_parsed = []
        for s in parsed:
            if '. ' in s:
                fully_parsed.extend(s.split('. '))
            else:
                fully_parsed.append(s)

        sentences = []
        for idx, s in enumerate(fully_parsed):
            tag = '{}_{}'.format(info, idx)
            sentences.append(Sentence(tag=tag, text=s))
            
        return sentences


class Corpus:
    def __init__(self):
        self.fdir = os.path.join(cfg['root'], cfg['fdir_corpus'])

    def load_single(self, **kwargs):
        if 'fpath' in kwargs.keys():
            fpath = kwargs.get('fpath')
            with open(fpath, 'rb') as f:
                return pk.load(f)

        elif 'fname' in kwargs.keys():
            fname = kwargs.get('fname')
            with open(os.path.join(self.fdir, fname), 'rb') as f:
                return pk.load(f)

        else:
            print('ERROR: Wrong FileName')
            return None

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


class Preprocessor:
    def __remove_trash_characters(self, text):
        text = text.lower()
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
                sents_cleaned.append(Sentence(tag=s.tag, text=cleaned_text))

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

        def __read_ngram_map():
            fpath_ngram_map = '/data/blank54/workspace/project/spec/corpus/thesaurus/ngram_map.txt'
            ngram_map = defaultdict(str)
            with open(fpath_ngram_map, 'r', encoding='utf-8') as f:
                for pair in f.read().strip().split('\n'):
                    l, r = pair.split('  ')
                    ngram_map[l] = r
            return ngram_map

        def __normalize_ngram(text, ngram_map):
            input_text = text
            for l, r in ngram_map.items():
                text = re.sub(l, r, text)
            if text != input_text:
                check = True
            else:
                check = False
            return text, check

        def __singularize(text):
            return ' '.join([singularize(w) for w in text.split()])

        if do:
            normalized_tokens = defaultdict(list)
            sents_normalized = []
            
            sents = kwargs.get('sents')
            re_num = re.compile('\d+\.*\d*\.*\d*')
            re_url = re.compile('.*www.*')

            unit_map = __read_unit_list()
            ngram_map = __read_ngram_map()

            with tqdm(total=len(sents)) as pbar:
                for s in sents:
                    tag, text = s.tag, s.text

                    # Numbers
                    if re_num.findall(text):
                        text = re_num.sub('NUM', text)
                        normalized_tokens['NUM'].append(tag)

                    # URLs
                    if re_url.findall(text):
                        text =re_url.sub('URL', text)
                        normalized_tokens['URL'].append(tag)

                    # Ngrams
                    text, check = __normalize_ngram(text, ngram_map)
                    if check:
                        normalized_tokens['NGRAM'].append(tag)
                    else:
                        pass

                    # Units
                    if any([w for w in text.split(' ') if w in unit_map]):
                        text = __normalize_unit(text, unit_map)
                        normalized_tokens['UNIT'].append(tag)

                    # Singularize
                    text = __singularize(text)

                    sents_normalized.append(Sentence(tag=tag, text=text))
                    pbar.update(1)

            with open(fpath, 'wb') as f:
                pk.dump((sents_normalized, normalized_tokens), f)

        else:
            with open(fpath, 'rb') as f:
                sents_normalized, normalized_tokens = pk.load(f)            

        return sents_normalized, normalized_tokens

    def pos_tagging(self, fpath, do, **kwargs):
        if do:
            sents = kwargs.get('sents')
            sents_pos = []
            with tqdm(total=len(sents)) as pbar:
                for s in sents:
                    s.pos = nltk.pos_tag(s.text.split())
                    sents_pos.append(s)
                    pbar.update(1)

            with open(fpath, 'wb') as f:
                pk.dump(sents_pos, f)

        else:
            with open(fpath, 'rb') as f:
                sents_pos = pk.load(f)

        return sents_pos

    def extract_sent_verb(self, fpath, do, **kwargs):
        if do:
            sents = kwargs.get('sents')
            verb_tag_list = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
            sents_verb = []
            with tqdm(total=len(sents)) as pbar:
                for s in sents:
                    if any([t for w, t in s.pos if t in verb_tag_list]):
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


class Embedding:
    def word2vec(self, fpath, train, **kwargs):
        if train:
            docs = kwargs.get('docs')
            parameters = kwargs.get('parameters')

            model = Word2Vec(
                size=parameters['size'],
                window=parameters['window'],
                min_count=parameters['min_count'],
                workers=parameters['workers'],
                sg=parameters['sg'],
                hs=parameters['hs'],
                negative=parameters['negative'],
                ns_exponent=parameters['ns_exponent'],
                iter=parameters['iter'],
            )
            model.build_vocab(sentences=docs)
            model.train(sentences=docs, total_examples=model.corpus_count, epochs=model.iter)

            w2v_model = Word2VecModel(docs=docs, model=model, parameters=parameters)
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            with open(fpath, 'wb') as f:
                pk.dump(w2v_model, f)

        else:
            with open(fpath, 'rb') as f:
                w2v_model = pk.load(f)

        return w2v_model


# class FlowMargin:
#     def 