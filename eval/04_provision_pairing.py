#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
import pandas as pd
from collections import defaultdict

from config import Config
with open('/data/blank54/workspace/project/spec/spec.cfg', 'r') as f:
    cfg = Config(f)

sys.path.append(cfg['root'])
from analysis import *
utils = Utils()


def show_relevance():
    fdir = os.path.join(cfg['root'], cfg['fdir_d2v_model'])
    fname = 'section_manual_ngram_500_10_10_10_1_5_200_1.pk'
    fpath = os.path.join(fdir, fname)

    with open(fpath, 'rb') as f:
        section_d2v_model = pk.load(f)

    target_section = 'Qatar_Qatar_2014_06_05'
    relevant_section_list = [
        'Australia_Tasmania_2017_04_09',
        'Australia_Tasmania_2017_04_21',
        'Qatar_Qatar_2010_06_04',
        'Qatar_Qatar_2010_06_05',
        'United States_Alabama_2018_04_08',
        'United States_Arkansas_2014_04_05',
        'United States_Arkansas_2014_04_06',
        'United States_Arkansas_2014_04_07',
        'United States_Arkansas_2014_04_14',
        'United States_Connecticut_2018_04_19'
        ]

    for s in relevant_section_list:
        target_vector = section_d2v_model.model.docvecs[target_section]
        relevant_vector = section_d2v_model.model.docvecs[s]
        relevance = utils.relevance(target_vector, relevant_vector)

        print(s)
        print('{:.03f}'.format(relevance))
        print('_______________________')

def show_casestudy_confusion_matrix():
    fdir = os.path.join(cfg['root'], cfg['fdir_result_paragraph_pairing_casestudy'])
    
    eval_dict = defaultdict(int)
    for fname in os.listdir(fdir):
        fpath = os.path.join(fdir, fname)

        data = pd.read_excel(fpath)
        eval_list = data['eval']
        for e in eval_list:
            eval_dict[e] += 1

    accuracy = utils.accuracy(tp=eval_dict['TP'], tn=eval_dict['TN'], fp=eval_dict['FP'], fn=eval_dict['FN'])
    print(eval_dict)
    print('Acccuracy: {:.02f}'.format(accuracy*100))


if __name__ == '__main__':
    show_casestudy_confusion_matrix()