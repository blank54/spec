#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
import itertools
import pandas as pd
from random import randrange
from collections import defaultdict

sys.path.append('/data/blank54/workspace/project/spec/')
from specutil import SpecPath
from analysis import Read, Write
specpath = SpecPath()
read = Read()
write = Write()


def prepare_pairs(TARGET_TAG, REFER_TAG):
    target_docs = read.docs_included(iter_unit='paragraph', hyper_tag=TARGET_TAG)
    refer_docs = read.docs_included(iter_unit='paragraph', hyper_tag=REFER_TAG)

    data = defaultdict(list)
    for target_doc, refer_doc in itertools.product(target_docs, refer_docs):
        data['left_tag'].append(target_doc.tag)
        data['left_text'].append(target_doc.text.lower())
        data['right_tag'].append(refer_doc.tag)
        data['right_text'].append(refer_doc.text.lower())
        data['label'].append(randrange(0,5,1))

    return data

def write_data_as_xlsx(data):
    data_df = pd.DataFrame(data)
    fname = 'L-{}_R-{}.xlsx'.format(TARGET_TAG, REFER_TAG)
    fdir = specpath.fdir_ppr_data_raw
    fpath = os.path.join(fdir, fname)

    with pd.ExcelWriter(path=fpath) as writer:
        write.makedir(path=fpath)
        data_df.to_excel(excel_writer=writer, index=False)
    print('Data Preparation for ProvisionPairing')
    print('    | fdir : ../{}'.format(fdir))
    print('    | fname: {}'.format(fname))

def import_exist_data(fdir, fname):
    fpath = os.path.join(fdir, fname)
    TARGET_TAG = '_'.join(fname.split('_')[:5]).lower()
    REFER_TAG = '_'.join(fname.replace('.xlsx', '').split('_')[5:]).lower()
    exist_data = pd.read_excel(fpath)
    return TARGET_TAG, REFER_TAG, exist_data

def export_data(data, TARGET_TAG, REFER_TAG):
    data_df = pd.DataFrame(data)
    fname = 'L-{}_R-{}.xlsx'.format(TARGET_TAG, REFER_TAG)
    fdir = specpath.fdir_ppr_data_raw
    fpath = os.path.join(fdir, fname)

    with pd.ExcelWriter(path=fpath) as writer:
        write.makedir(path=fpath)
        data_df.to_excel(excel_writer=writer, index=False)
    print('Data Preparation for ProvisionPairing')
    print('    | fdir : ../{}'.format(fdir))
    print('    | fname: {}'.format(fname))

def convert_from_exiting_data():
    fdir = specpath.fdir_ppr_data_exist
    for fname in os.listdir(fdir):
        TARGET_TAG, REFER_TAG, exist_data = import_exist_data(fdir, fname)
        target_docs = read.docs_included(iter_unit='paragraph', hyper_tag=TARGET_TAG)
        refer_docs = read.docs_included(iter_unit='paragraph', hyper_tag=REFER_TAG)

        data = defaultdict(list)
        for _, record in exist_data.iterrows():
            if record['eval'] == 'FN':
                continue
            elif record['eval'] == 'TN':
                for doc in refer_docs:
                    data['left_tag'].append(record['target'].lower())
                    data['left_text'].append([doc for doc in target_docs if doc.tag == record['target'].lower()][0].text)
                    data['right_tag'].append(doc.tag)
                    data['right_text'].append(doc.text)
                    data['label'].append(str(0))
            else:
                data['left_tag'].append(record['target'].lower())
                data['left_text'].append([doc for doc in target_docs if doc.tag == record['target'].lower()][0].text)
                data['right_tag'].append(record['relevant'].lower())
                data['right_text'].append([doc for doc in refer_docs if doc.tag == record['relevant'].lower()][0].text)
                
                if record['eval'] == 'TP':
                    data['label'].append(str(4))
                elif record['eval'] == 'FP':
                    data['label'].append(str(0))

        export_data(data, TARGET_TAG, REFER_TAG)



if __name__ == '__main__':
    TARGET_TAG = 'qatar_qatar_2014_06_05'
    REFER_TAG = 'australia_tasmania_2017_04_09'

    ## Prepare new data
    # data = prepare_pairs(TARGET_TAG, REFER_TAG)
    # write_data_as_xlsx(data)

    ## Utilize already paired data based on Doc2Vec
    convert_from_exiting_data()