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
from analysis import *
preprocessor = Preprocessor()
ngram_parser = NgramParser()
read = Read()
write = Write()


def preprocess(iter_unit, do):
    size_before = 0
    size_after = 0

    unit_map = read.word_map(option='unit')
    stopword_list = read.stopword_list()

    fdir_corpus = os.path.join(cfg['root'], cfg['fdir_corpus'], iter_unit)
    with tqdm(total=len(os.listdir(fdir_corpus))) as pbar:
        for d in read.docs(iter_unit=iter_unit):
            size_before += os.path.getsize(d.fpath)

            if do:
                text_normalized = preprocessor.normalize(text=d.text, unit_map=unit_map)
                d.token = preprocessor.tokenize(text=text_normalized)
                d.stop = preprocessor.stopword_removal(sent=d.token, stopword_list=stopword_list)
                d.stem = preprocessor.stemmize(sent=d.stop)
                d.lemma = preprocessor.lemmatize(sent=d.stop)
                write.object(obj=d, fpath=d.fpath)
            else:
                pass

            size_after += os.path.getsize(d.fpath)
            pbar.update(1)

    print('Preprocessing [{}]\n └ {:,.02f} MB -> {:,.02f} MB'.format(iter_unit, size_before/1024**2, size_after/1024**2))


## Ngram
def ngram_parsing(iter_unit, do):
    size_before = 0
    size_after = 0

    if do:
        docs = [d.lemma for d in read.docs(iter_unit=iter_unit)]
        bigram_counter = ngram_parser.count(docs=docs)
    else:
        pass

    fdir_corpus = os.path.join(cfg['root'], cfg['fdir_corpus'], iter_unit)
    with tqdm(total=len(os.listdir(fdir_corpus))) as pbar:
        for d in read.docs(iter_unit=iter_unit):
            size_before += os.path.getsize(d.fpath)

            if do:
                d.ngram = ngram_parser.parse(sent=d.lemma, bigram_counter=bigram_counter)
                write.object(obj=d, fpath=d.fpath)
            else:
                pass

            size_after += os.path.getsize(d.fpath)
            pbar.update(1)

        print('Ngram Parsing [{}]:\n └ {:,.02f} MB -> {:,.02f} MB'.format(iter_unit, size_before/1024**2, size_after/1024**2))





if __name__ == '__main__':
    preprocess(iter_unit='section_manual', do=True)
    ngram_parsing(iter_unit='section_manual', do=True)

    # preprocess(iter_unit='paragraph', do=True)
    # ngram_parsing(iter_unit='paragraph', do=True)







# ## Data Import
# fdir_sents = os.path.join(cfg['root'], cfg['fdir_corpus'], 'sentence/')
# fpath_sents = os.path.join(fdir_sents, 'sentences_original.pk')
# sents_original = corpus.load_single(fpath=fpath_sents)
# print('# of Original Corpus: {:,}'.format(len(sents_original)))


# ## Text Cleaning
# fpath_sents_cleaned = os.path.join(fdir_sents, 'sentences_cleaned.pk')
# sents_cleaned = preprocessor.text_cleaning(sents=sents_original, 
#                                            fpath=fpath_sents_cleaned, 
#                                            do=False)
# print('Size of Original: {:,.02f} MB'.format(os.path.getsize(fpath_sents)/1024**2))
# print('Size of Cleaned: {:,.02f} MB'.format(os.path.getsize(fpath_sents_cleaned)/1024**2))


# ## Normalization
# fpath_sents_normalized = os.path.join(fdir_sents, 'sentences_normalized.pk')
# sents_normalized, normalized_tokens = preprocessor.normalize(sents=sents_cleaned, 
#                                                              fpath=fpath_sents_normalized,
#                                                              do=False)
# print('# of Sentences with Replaced Tokens: {:,}'.format(len(sents_normalized)))
# for t in normalized_tokens.keys():
#     print(' └ # of Sentences Operated with {}: {:,}'.format(t, len(normalized_tokens[t])))

# singularized, cnt_before, cnt_after = stat.singularized(sents_cleaned)
# print('# of Words: {:,}'.format(cnt_before))
# print('# of Singularized Words: {:,}'.format(cnt_after))


# ## PoS Tagging
# fpath_sents_pos = os.path.join(fdir_sents, 'sentences_pos.pk')
# sents_pos = preprocessor.pos_tagging(sents=sents_normalized,
#                                      fpath=fpath_sents_pos,
#                                      do=False)
# print('# of PoS tags: {:,}'.format(len(list(itertools.chain(*[s.pos for s in sents_pos])))))



# ## Extract Verbal Sentences
# fpath_sents_verbal = os.path.join(fdir_sents, 'sentences_verbal.pk')
# sents_verbal = preprocessor.extract_sent_verb(sents=sents_pos, 
#                                               fpath=fpath_sents_verbal, 
#                                               do=False)
# print('# of Original Sents: {:,}'.format(len(sents_pos)))
# print('# of Sents with Verb: {:,}'.format(len(sents_verbal)))


# ## Stopword Removal
# '''
# Refer to other studies that utilized Word2Vec
# '''
# fpath_sents_wo_stop = os.path.join(fdir_sents, 'sentences_wo_stop.pk')
# sents_wo_stop = preprocessor.stopword_removal(sents=sents_verbal,
#                                               fpath=fpath_sents_wo_stop,
#                                               do=False)
# print('# of Sents without Stopwords: {:,}'.format(len(sents_wo_stop)))
# print(' └ # of Words BEFORE Stopword Removal: {:,}'.format(len(list(itertools.chain(*[s.pos for s in sents_verbal])))))
# print(' └ # of Words AFTER Stopword Removal : {:,}'.format(len(list(itertools.chain(*[s.pos for s in sents_wo_stop])))))

# ## Text Chunking
# fpath_sents_chunk = os.path.join(fdir_sents, 'sentences_chunk.pk')
# sents_chunk = preprocessor.chunking(sents=sents_wo_stop,
#                                     fpath=fpath_sents_chunk,
#                                     do=False)
# print('# of Chunks: {:,}'.format(len(list(itertools.chain(*[s.chunk for s in sents_chunk])))))