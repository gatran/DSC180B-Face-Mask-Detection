#!/usr/bin/env python

import os
import sys
import json
import shutil

data_ingest_params = 'config/data-params.json'

def load_params(fp):
    with open(fp) as fh:
        param = json.load(fh)
    return param

def main(targets):
    if 'clean' in targets:
        shutil.rmtree('results/', ignore_errors=True)
        
    if "test" in targets:
        os.makedirs('results')
        os.system("python " + load_params(data_ingest_params)["notebook_path"] + " --image-path "+ load_params(data_ingest_params)["img"])

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)

