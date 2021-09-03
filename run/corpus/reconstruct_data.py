import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-2])
sys.path.append(rootpath)

from specutil import SpecPath
specpath = SpecPath()

import pickle as pk
import pandas as pd
from collections import defaultdict


if __name__ == '__main__':
    flist = os.listdir(specpath.fdir_corpus_paragraph)
    data = defaultdict(list)

    for fname in flist:
        with open(os.path.join(specpath.fdir_corpus_paragraph, fname), 'rb') as f:
            paragraph = pk.load(f)
            data['tag'].append(paragraph.tag)
            data['text'].append(paragraph.text)

    df = pd.DataFrame(data)

    fname_data = 'paragraph.xlsx'
    fpath_data = os.path.join(specpath.fdir_data_paragraph, fname_data)
    df.to_excel(fpath_data)