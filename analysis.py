#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import re
import sys
import csv
import copy
import operator
import itertools
import numpy as np
import pickle as pk
from time import time
from tqdm import tqdm
from collections import defaultdict, Counter

import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.metrics.pairwise import cosine_similarity
from keras.preprocessing.sequence import pad_sequences

import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
matplotlib.rc('font', family='NanumBarunGothic')

from config import Config
with open('/data/blank54/workspace/project/spec/spec.cfg', 'r') as f:
    cfg = Config(f)

sys.path.append(cfg['root'])
from object import Section, Paragraph, LabeledSentence, NER_Labels, NER_Corpus


class Utils:
    def parse_fname(self, fpath, iter_unit):
        fname = os.path.basename(fpath)
        if iter_unit == 'spec':
            return '_'.join(fname[:-4].split('_')[:3])
        elif iter_unit == 'section' or iter_unit == 'section_manual':
            return '_'.join(fname[:-4].split('_')[:5])

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

    def relevance(self, vec1, vec2):
        return (cosine_similarity([vec1], [vec2])[0][0]+1)/2 # normalized to [0,1]

    def pairing(self, tag1_list, tag2_list):
        pairs = defaultdict(dict)
        for tag1, vec1 in tag1_list:
            for tag2, vec2 in tag2_list:
                if tag2 == tag1:
                    continue
                else:
                    score = self.relevance(vec1, vec2)
                    pairs[tag1][tag2] = score
                    # pairs[tag2][tag1] = score

        sorted_pairs = defaultdict(list)
        for tag, paired_tags in pairs.items():
            sorted_pairs[tag] = list(sorted(paired_tags.items(), key=lambda x:x[1], reverse=True))

        return sorted_pairs

    def assign_word_index(self, docs, **kwargs):
        add = kwargs.get('add', [])
        words = list(set(itertools.chain(*[doc.words for doc in docs])))
        for w in add:
            words.append(w)

        word2id = {w: i for i, w in enumerate(words)}
        id2word = {i: w for i, w in enumerate(words)}
        return word2id, id2word

    def accuracy(self, tp, tn, fp, fn):
        return (tp+tn)/(tp+tn+fp+fn)

    def f1_score(self, p, r):
        if p != 0 or r != 0:
            return (2*p*r)/(p+r)
        else:
            return 0

    def f1_score_from_matrix(self, matrix, round=3):
        f1_list = []
        matrix_size = len(matrix)
        for i in range(matrix_size):
            corr = matrix[i, i]
            pred = matrix[matrix_size-1, i]
            real = matrix[i, matrix_size-1]

            precision = corr/max(pred, 1)
            recall = corr/max(real, 1)
            f1_list.append(self.f1_score(precision, recall))

        f1_average = np.mean(f1_list).round(3)
        return f1_list, f1_average


class IO:
    fdir_model = os.path.join(cfg['root'], cfg['fdir_model'])
    fdir_corpus = os.path.join(cfg['root'], cfg['fdir_corpus'])
    fdir_thesaurus = os.path.join(cfg['root'], cfg['fdir_thesaurus'])
    fdir_ner_labels = os.path.join(cfg['root'], cfg['fdir_ner_labels'])
    fdir_ner_corpus = os.path.join(cfg['root'], cfg['fdir_ner_corpus'])


class Read(IO):
    def docs(self, iter_unit):
        fdir = os.path.join(self.fdir_corpus, iter_unit)
        for fname in os.listdir(fdir):
            fpath = os.path.join(fdir, fname)
            with open(fpath, 'rb') as f:
                yield pk.load(f)

    def docs_included(self, iter_unit, hyper_tag):
        return [d for d in self.docs(iter_unit=iter_unit) if hyper_tag.lower() in d.tag]

    def object(self, format, fpath):
        if format == 'pk':
            with open(fpath, 'rb') as f:
                return pk.load(f)

    def word_map(self, option):
        fpath = os.path.join(self.fdir_thesaurus, '{}.txt'.format(option))
        word_map = defaultdict(str)
        with open(fpath, 'r', encoding='utf-8') as f:
            for pair in f.read().strip().split('\n'):
                l, r = pair.split('  ')
                word_map[l] = r
        return word_map

    def stopword_list(self):
        fpath = os.path.join(self.fdir_thesaurus, 'stopword_list.txt')
        with open(fpath, 'r', encoding='utf-8') as f:
            return [word.strip() for word in f.read().strip().split('\n') if word]

    def ner_labels(self):
        fpath = os.path.join(self.fdir_ner_labels, 'ner_labels.txt')
        with open(fpath, 'r', encoding='utf-8') as f:
            label_list = [tuple(pair.split('  ')) for pair in f.read().strip().split('\n')]

        label_list.append(('__PAD__', '6'))
        label_list.append(('__UNK__', '7'))
        return NER_Labels(label_list=label_list)

    def ner_weighted_labels(self):
        fpath = os.path.join(self.fdir_ner_labels, 'ner_weighted_labels.txt')
        with open(fpath, 'r', encoding='utf-8') as f:
            return [l.strip() for l in f.read().strip().split('\n') if l]

    def ner_labeled_doc(self, fpath):
        with open(fpath, 'r', encoding='utf-8') as f:
            lines = [line for line in csv.reader(f)]

        doc = []
        errors = []
        if len(lines) % 2 == 1:
            print('Error: len(sent) != len(labels)')

        else:
            for idx in range(len(lines)):
                if idx % 2 == 0:
                    section_info = Utils().parse_fname(fpath=fpath, iter_unit='section')
                    line_num = int(idx/2)
                    tag = '{}_{}'.format(section_info, line_num)
                    words = [w.lower() for w in lines[idx] if w]
                else:
                    labels = [l for l in lines[idx] if l]
                    if len(words) == len(labels):
                        doc.append(LabeledSentence(tag=tag, words=words, labels=labels))
                    else:
                        errors.append((fpath, idx))

        return doc, errors

    def ner_labeled_docs(self):
        fdir = os.path.join(cfg['root'], cfg['fdir_data_ner'])
        docs = []
        for fname in os.listdir(fdir):
            fpath = os.path.join(fdir, fname)
            doc, errors = self.ner_labeled_doc(fpath=fpath)
            docs.extend(doc)
            if errors:
                print(errors)

        for sent in docs:
            yield sent

    def ner_corpus(self, **kwargs):
        if 'fname' in kwargs.keys():
            fname = kwargs.get('fname')
            fpath = os.path.join(cfg['root'], cfg['fdir_ner_corpus'], fname)
        elif 'fpath' in kwargs.keys():
            fpath = kwargs.get('fpath')

        with open(fpath, 'rb') as f:
            return pk.load(f)


class Write(IO):
    def makedir(self, path):
        if path.endswith('/'):
            os.makedirs(path, exist_ok=True)
        else:
            os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)

    def object(self, obj, fpath):
        self.makedir(fpath)
        with open(fpath, 'wb') as f:
            pk.dump(obj, f)


class BuildCorpus(IO):
    def section(self, tag, section_text):
        sents = []
        for p in self.section2paragraph(section_text=section_text):
            if p.text:
                sents.append(p.text)
            else:
                continue

        return Section(tag=tag, text='  '.join(sents), sents=sents)


    def section2paragraph(self, section_text):
        for paragraph in section_text.split('\n\n'):
            lines = paragraph.split('\n')
            tag = Utils().line2paragraph_info(lines[0])
            text = '  '.join(lines[1:])
            fpath = ''
            yield Paragraph(tag=tag, text=text)

    def ner_corpus(self, max_sent_len, fname, build):
        fdir = os.path.join(self.fdir_ner_corpus)
        fpath = os.path.join(fdir, fname)

        if build:
            docs = list(Read().ner_labeled_docs())
            ner_labels = Read().ner_labels()
            weighted_labels = Read().ner_weighted_labels()

            add = ['__PAD__', '__UNK__']
            word2id, id2word = Utils().assign_word_index(docs=docs, add=add)

            X_words = []
            Y_labels = []
            for doc in docs:
                X_words.append([word2id[w] for w in doc.words])
                Y_labels.append(doc.labels)
                if any(w in [ner_labels.id2label[int(l)] for l in doc.labels] for w in weighted_labels):
                    X_words.append([word2id[w] for w in doc.words])
                    Y_labels.append(doc.labels)

            X_words_pad = pad_sequences(
                maxlen=max_sent_len,
                sequences=X_words,
                padding='post',
                value=word2id['__PAD__'])
            Y_labels_pad = pad_sequences(
                maxlen=max_sent_len,
                sequences=Y_labels,
                padding='post',
                value=ner_labels.label2id['__PAD__'])

            ner_corpus = NER_Corpus(docs=docs, ner_labels=ner_labels, weighted_labels=weighted_labels, 
                max_sent_len=max_sent_len,
                word2id=word2id, id2word=id2word, X_words=X_words, Y_labels=Y_labels, 
                X_words_pad=X_words_pad, Y_labels_pad=Y_labels_pad, fpath=fpath)
            
            Write().object(obj=ner_corpus, fpath=fpath)

        else:
            ner_corpus = Read().ner_corpus(fname=fname)

        return ner_corpus


class Preprocessor:
    def __init__(self, **kwargs):
        self.stemmer = LancasterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def __separate_sentence(self, text):
        sents = []
        for sent in text.split('  '):
            sent_for_split = re.sub('\.(?=[A-Zㄱ-힣])', '  ', sent).split('  ')
            if all((len(s.split(' '))>=5 for s in sent_for_split)):
                sents.extend(sent_for_split)
            else:
                sents.append(sent)
        return sents

    def __remove_trash_characters(self, text):
        text = text.lower()
        text = re.sub('[^ \'\?\./0-9a-zA-Zㄱ-힣\n]', '', text)

        text = text.replace(' / ', '/')
        text = re.sub('\.+\.', ' ', text)
        text = text.replace('\\\\', '\\').replace('\\r\\n', '')

        text = text.replace('\n', '  ')
        text = re.sub('\. ', '  ', text)
        text = re.sub('\s+\s', ' ', text).strip()
        
        if text.endswith('\.'):
            text = text[:-1]
            
        return text

    def __marking(self, sent):
        for i, w in enumerate(sent):
            if re.match('www.', str(w)):
                sent[i] = 'LINK'
            elif re.search('\d+\d\.\d+', str(w)):
                sent[i] = 'REF'
            elif re.match('\d', str(w)):
                sent[i] = 'NUM'
        return sent

    def __sentence_hierarchy(self, sents):
        if not any((s.startswith('?') for s in sents)):
            return sents

        sents_with_sub = []
        origin = ''
        origin2 = ''
        for i in range(len(sents)):
            if not sents[i].startswith('?'):
                sents_with_sub.append(sents[i])
                origin = sents[i]
            else:
                if not sents[i].startswith('??'):
                    sents_with_sub.append(origin+sents[i])
                    origin2 = sents[i]
                else:
                    sents_with_sub.append(origin+origin2+sents[i])
        del origin, origin2
        return [s.replace('?', '') for s in sents_with_sub]

    def __replace_unit(self, sent, unit_map):
        sent = ' '+sent.strip()+' '
        for l, r in unit_map.items():
            sent = sent.replace(' {} '.format(str(l)), ' {} '.format(str(r))).strip()
        return sent

    def normalize(self, text, unit_map):
        sents = []
        for s in self.__separate_sentence(text):
            text_cleaned = self.__remove_trash_characters(s)
            text_marked = ' '.join(self.__marking(text_cleaned.split(' ')))
            sents.append(text_marked)

        sents_normalized = []
        for s in self.__sentence_hierarchy(sents):
           sents_normalized.append(self.__replace_unit(s, unit_map))

        return '  '.join(sents_normalized)

    def tokenize(self, text):
        unigrams = [w for w in re.split(' |  |\n', text) if len(w)>0]
        return unigrams

    def stopword_removal(self, sent, stopword_list):
        return [w for w in sent if w not in stopword_list]

    def stemmize(self, sent):
        return [self.stemmer.stem(w) for w in sent]

    def lemmatize(self, sent):
        return [self.lemmatizer.lemmatize(w) for w in sent]

    def __doc2bigrams(self, doc):
        bigrams = []
        for idx in range(0, len(doc)):
            bigrams.append('-'.join(doc[idx:idx+2]))
        return bigrams

    def count(self, docs, **kwargs):
        return Counter(itertools.chain(*[self.__doc2bigrams(doc) for doc in docs]))

    def parse(self, sent, bigram_counter, min_count=20):
        bigram_list = [bigram for bigram, cnt in bigram_counter.items() if cnt >= min_count]
        
        sent_with_ngram = []
        switch = False
        ngram = [sent[0]]

        for i in range(1, len(sent)):
            bigram = '{}-{}'.format(sent[i-1], sent[i])

            if bigram in bigram_list:
                ngram.append(sent[i])
                switch = True
            else:
                sent_with_ngram.append('-'.join(ngram))
                ngram = [sent[i]]
                switch = False

            if i == len(sent)-1 and switch == True:
                sent_with_ngram.append('-'.join(ngram))
            else:
                continue

        return sent_with_ngram


class Visualizer:
    def network(self, data):
        graph = nx.DiGraph()
        for node_from, node_to, score in data:
            graph.add_edge(node_from, node_to, weight=score)
        pos = nx.spring_layout(graph, k=0.08)

        fig, ax = plt.subplots(figsize=(10, 8), dpi=600)
        nx.draw_networkx(
            graph, pos,
            arrows=True,
            node_size=10,
            font_size=0,
            width=0.6,
            edge_color='grey',
            node_color='purple',
            with_labels=False,
            ax=ax)

        for key, value in pos.items():
            ax.text(x=value[0],
                    y=value[1]+0.025,
                    s=key,
                    bbox=dict(facecolor='white', alpha=0, edgecolor='white'),
                    horizontalalignment='center',
                    fontsize=8)

        plt.show()