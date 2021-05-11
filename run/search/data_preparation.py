#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
import itertools
import pandas as pd
from collections import defaultdict

sys.path.append('/data/blank54/workspace/project/spec/')
from specutil import SpecPath
from analysis import Read, Write
specpath = SpecPath()
read = Read()
write = Write()


def build_sent_pair(TARGET_TAG, REFER_TAG):
    target_sents = read.docs_included(iter_unit='sentence', hyper_tag=TARGET_TAG)
    refer_sents = read.docs_included(iter_unit='sentence', hyper_tag=REFER_TAG)

    data = defaultdict(list)
    for target_sent, refer_sent in itertools.product(target_sents, refer_sents):
        data['left_tag'].append(target_sent.tag)
        data['left_text'].append(target_sent.text.lower())
        data['right_tag'].append(refer_sent.tag)
        data['right_text'].append(refer_sent.text.lower())
        data['label'].append(str(0))

    return data

def write_data_as_xlsx(data):
    global sample_size

    data_df = pd.DataFrame(data)
    fname = 'L-{}_R-{}_SAMPLE-{}.xlsx'.format(TARGET_TAG, REFER_TAG, sample_size)
    fdir = specpath.fdir_search_data
    fpath = os.path.join(fdir, fname)

    with pd.ExcelWriter(path=fpath) as writer:
        write.makedir(path=fpath)
        data_df[:sample_size].to_excel(excel_writer=writer, index=False)
    print('Data Preparation for ProvisionPairing')
    print('    | fdir : ../{}'.format(fdir))
    print('    | fname: {}'.format(fname))


if __name__ == '__main__':
    sample_size = 300
    for target_fname, refer_fname in itertools.combinations(os.listdir(specpath.fdir_data_section_manual), r=2):
        TARGET_TAG = target_fname.replace('.txt', '').lower()
        REFER_TAG = refer_fname.replace('.txt', '').lower()
        
        data = build_sent_pair(TARGET_TAG, REFER_TAG)
        write_data_as_xlsx(data)