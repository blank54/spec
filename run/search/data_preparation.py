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


def build_sent_pair(TARGET_TAG, REFER_TAG):
    target_sents = read.docs_included(iter_unit='sentence', hyper_tag=TARGET_TAG)
    refer_sents = read.docs_included(iter_unit='sentence', hyper_tag=REFER_TAG)

    data = defaultdict(list)
    for target_sent, refer_sent in itertools.product(target_sents, refer_sents):
        print(target_sent.tag)
        print(refer_sent.tag)
    #     data['left_tag'].append(target_sent.tag)
    #     data['left_text'].append(target_sent.text.lower())
    #     data['right_tag'].append(refer_sent.tag)
    #     data['right_text'].append(refer_sent.text.lower())
    #     data['label'].append(randrange(0,5,1))

    # return data

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


if __name__ == '__main__':
    TARGET_TAG = 'qatar_qatar_2014_06_05'
    REFER_TAG = 'australia_tasmania_2017_04_09'

    build_sent_pair(TARGET_TAG, REFER_TAG)