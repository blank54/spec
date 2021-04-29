#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
import itertools
import pandas as pd
from collections import defaultdict

from config import Config
with open('/data/blank54/workspace/project/spec/spec.cfg', 'r') as f:
    cfg = Config(f)

sys.path.append(cfg['root'])
from analysis import Read, Write
read = Read()
write = Write()

from random import randrange


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
    fdir = os.path.join(cfg['root'], cfg['fdir_data_provision_pair_raw'])
    fpath = os.path.join(fdir, fname)

    with pd.ExcelWriter(path=fpath) as writer:
        write.makedir(path=fpath)
        data_df.to_excel(excel_writer=writer, index=False)
    print('Data Preparation for ProvisionPairing')
    print('    | fdir : ../{}'.format(cfg['fdir_data_provision_pair_raw']))
    print('    | fname: {}'.format(fname))


if __name__ == '__main__':
    TARGET_TAG = 'qatar_2014_06_05'
    REFER_TAG = 'qatar_2010_06_05'

    data = prepare_pairs(TARGET_TAG, REFER_TAG)
    write_data_as_xlsx(data)