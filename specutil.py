#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os


class SpecPath:
    root = '/data/blank54/workspace/project/spec/'

    ## Provision Pairing
    fdir_ppr_data_exist = os.path.join(root, 'result/paragraph_pairing_casestudy_eval/')
    fdir_ppr_data_raw = os.path.join(root, 'data/provision_pair/raw/')