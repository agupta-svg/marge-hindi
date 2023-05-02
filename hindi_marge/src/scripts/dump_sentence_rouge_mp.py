# -*- coding: utf-8 -*-
import pdb
import os
from os.path import dirname, abspath, join, exists
from pathlib import Path
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))
import logging

import json
from tqdm import tqdm
import shutil
from multiprocessing import Pool
from pyrouge import Rouge155
from indicnlp.tokenize import sentence_tokenize


import utils.config_loader as config
from utils.config_loader import config_meta, path_parser, logger
# import utils.tools as tools
# from summ import compute_rouge
from data.hindi_parser import MultiNewsParser
# from data.cnndm_parser import CnnDmParser

# pdb.set_trace()
if config.path_type == 'afs':
    DP_PROJ = Path('/disk/scratch/margesum')
    N_PROC = 30
else:
    DP_PROJ = Path('./')
    N_PROC = 5
    

# DP_RAW_MN = DP_DATA / 'raw_mn'
DP_TEMP = DP_PROJ / 'temp'
DP_DATA = DP_PROJ / 'data'
DP_SENTENCE_MN = DP_DATA / 'sentence_mn'
DP_SENTENCE_CNNDM = DP_DATA / 'cnndm' / 'rouge_cnndm_1'
DP_SENTENCE_HINDI = DP_DATA/'hindi'/'rouge_hindi_1'


DATASET_VARS = ['train', 'val', 'test', 'debug']
SENTENCE_SPLIT_DICT = {"hi": sentence_tokenize.sentence_split}

HINDI_DATASET_SIZE = {
    'train': 51774,
    'val': 17259,
    'test': 17265,
}

# CNNDM_DATASET_SIZE = {
#     'train': 287227,
#     'val': 13368,
#     'test': 11490,
# }

"""
    Multi-process script for computing and dumping ROUHE scores for sentences from MultiNews or CNN/DailyMail.
"""

def _mp_get_rouge_handler(dp):
    if config.path_type == 'afs':  # copied from 2007 with removal of "-l 250"
        rouge_args = f'-a -n 2 -m -c 95 -r 1000 -f A -p 0.5 -t 0 -d -e {path_parser.afs_rouge_dir} -x'
        r = Rouge155(rouge_dir=str(path_parser.remote_root / 'ROUGE-1.5.5'),
                     rouge_args=rouge_args, 
                     log_level=logging.WARNING, 
                     config_parent_dir=str(path_parser.remote_root))
    else:
        #set local path 
        path_parser.local_rouge_dir= "/Users/aashigupta/Library/Caches/pypoetry/virtualenvs/hindi-marge-WsGnlyNd-py3.9/lib/python3.9/site-packages/pyrouge-0.1.3-py3.9.egg/pyrouge/tools/ROUGE-1.5.5/data"
        # rouge_args = '-a -n 2 -m -c 95 -r 1000 -f A -p 0.5 -t 0 -d -x'  # standard nist eval
        rouge_args = f'-a -n 2 -g -c 95 -r 1000 -f A -p 0.5 -t 0 -d -e {path_parser.local_rouge_dir} -x'
        #rouge_args = f'-a -n 2 -m -c 95 -r 1000 -f A -p 0.5 -t 0 -d -e /Users/aashigupta/Library/Caches/pypoetry/virtualenvs/hindi-marge-WsGnlyNd-py3.9/lib/python3.9/site-packages/pyrouge-0.1.3-py3.9.egg/pyrouge/tools/ROUGE-1.5.5/data -x'

        r = Rouge155(rouge_args=rouge_args)
    r.system_dir = dp
    r.model_dir = dp

    gen_sys_file_pat = '(\w*).sentence'
    gen_model_file_pat = '#ID#.summary'
    r.system_filename_pattern = gen_sys_file_pat
    r.model_filename_pattern = gen_model_file_pat

    return r


def mp_core_doc(arg):
    sid, sentence, summary = arg
    root = DP_TEMP / f'{sid}'
    if not exists(root):
        os.mkdir(root)
    
    sent_fp = root / f'{sid}.sentence'
    summary_fp = root / f'{sid}.summary'
    open(sent_fp, 'a').write(sentence)
    open(summary_fp, 'a').write(summary)

    rouge_handler = _mp_get_rouge_handler(dp=root)
    output = rouge_handler.convert_and_evaluate_language(language="hi",split_sentences=True)
    # output = rouge_handler.convert_and_evaluate(split_sentences=True)

    res = rouge_handler.output_to_dict(output)
    print(res)
    data = {
        'sid': sid,
        'sentence': sentence,
        'rouge_1_recall': res['rouge_1_recall'],
        'rouge_2_recall': res['rouge_2_recall'],
        'rouge_1_f1': res['rouge_1_f_score'],
        'rouge_2_f1': res['rouge_2_f_score'],
    }
    
    json_str = json.dumps(data, ensure_ascii=False)
    shutil.rmtree(root)
    return json_str

def mp_dump_sentence_mn(dataset_var, start=0):
    if dataset_var not in DATASET_VARS:
        raise ValueError(f'Illegal {dataset_var}!')

    if not exists(DP_TEMP):
        os.mkdir(DP_TEMP)

    mn_parser = MultiNewsParser(dataset_var=dataset_var)

    dump_fp = DP_SENTENCE_MN / f'{dataset_var}.json'
    print(f'Ready to dump json data to {dump_fp}')

    sample_generator = mn_parser.sample_generator()
    
    with open(dump_fp, 'a') as f:
        for cluster_idx, cluster_sentences, summary in tqdm(sample_generator, total=MN_DATASET_SIZE[dataset_var]):
            if cluster_idx < start:
                continue
            cluster_jsons = []
            for doc_idx, doc_sentences in enumerate(cluster_sentences):
                params = [
                    [f'{cluster_idx}_{doc_idx}_{sent_idx}', sent, summary] for sent_idx, sent in enumerate(doc_sentences)
                ]
                p = Pool(N_PROC)
                doc_jsons = p.map(mp_core_doc, params)
                if doc_jsons:
                    cluster_jsons.extend(doc_jsons)
            f.write('\n'.join(cluster_jsons)+'\n')

    os.rmdir(DP_TEMP)


# def mp_dump_sentence_cnndm(dataset_var, start=0):
#     if dataset_var not in DATASET_VARS:
#         raise ValueError(f'Illegal {dataset_var}!')

#     if not exists(DP_TEMP):
#         os.mkdir(DP_TEMP)

#     parser = CnnDmParser(dataset_var=dataset_var)
#     sample_generator = parser.sample_generator()

#     dump_fp = DP_SENTENCE_CNNDM / f'{dataset_var}.json'
#     logger.info((f'Ready to dump json data to {dump_fp}'))
    
#     with open(dump_fp, 'a') as f:
#         for doc_idx, doc_sentences, summary in tqdm(sample_generator, total=CNNDM_DATASET_SIZE[dataset_var]):
#             if doc_idx < start:
#                 continue
#             p = Pool(N_PROC)
#             params = [
#                 [f'{doc_idx}_{sent_idx}', sent, summary] for sent_idx, sent in enumerate(doc_sentences)
#             ]
#             doc_jsons = p.map(mp_core_doc, params)
#             f.write('\n'.join(doc_jsons)+'\n')
#     os.rmdir(DP_TEMP)


def check_disk(min_disk=20):
    available = float(list(shutil.disk_usage("./"))[-1]) / (1024 ** 3)
    print(available)
    return available > min_disk


# def gathered_mp_dump_sentence_mn(dataset_var, start=0, min_disk=20):
#     pdb.set_trace()
#     if not exists(DP_TEMP):
#         os.mkdir(DP_TEMP)

#     # Have to change this code for hindi
#     mn_parser = MultiNewsParser(dataset_var=dataset_var)
#     if dataset_var not in DATASET_VARS:
#         raise ValueError(f'Illegal {dataset_var}!')

#     if start == 0:
#         dump_fp = DP_SENTENCE_MN / f'{dataset_var}.json'
#     else:
#         dump_fp = DP_SENTENCE_MN / f'{dataset_var}_{start}.json'
    
#     print(f'Ready to dump json data to {dump_fp}')

#     sample_generator = mn_parser.sample_generator()

#     p = Pool(N_PROC)
#     with open(dump_fp, 'a') as f:
#         for cluster_idx, cluster_sentences, summary in tqdm(sample_generator, total=MN_DATASET_SIZE[dataset_var]):
#             if cluster_idx < start:
#                 continue

#             if cluster_idx % 50 == 0 and not check_disk(min_disk=min_disk):
#                 print(f'Disk less than {min_disk}. Break!')
#                 break

#             cluster_params = [[f'{cluster_idx}_{doc_idx}_{sent_idx}', sent, summary]
#                 for doc_idx, doc_sentences in enumerate(cluster_sentences)
#                 for sent_idx, sent in enumerate(doc_sentences)
#             ]
#             cluster_jsons = p.map(mp_core_doc, cluster_params)
#             f.write('\n'.join(cluster_jsons)+'\n')
#     p.close()
#     p.join()
#     os.rmdir(DP_TEMP)

def gathered_mp_dump_sentence_hindi(dataset_var, start=0, min_disk=20):

    if not exists(DP_TEMP):
        os.mkdir(DP_TEMP)
    if not exists(DP_SENTENCE_HINDI):
        os.mkdir(DP_SENTENCE_HINDI)    

    hindi_parser = MultiNewsParser(dataset_var=dataset_var)

    if dataset_var not in DATASET_VARS:
        raise ValueError(f'Illegal {dataset_var}!')

    if start == 0:
        dump_fp = DP_SENTENCE_HINDI / f'{dataset_var}.json'
    else:
        dump_fp = DP_SENTENCE_HINDI / f'{dataset_var}_{start}.json'
    
    print(f'Ready to dump json data to {dump_fp}')

    sample_generator = hindi_parser.sample_generator()


    p = Pool(N_PROC)
    with open(dump_fp, 'a') as f:
        for cluster_idx, cluster_sentences, summary in tqdm(sample_generator, total=HINDI_DATASET_SIZE[dataset_var]):
            if cluster_idx < start:
                continue

            if cluster_idx % 50 == 0 and not check_disk(min_disk=min_disk):
                print(f'Disk less than {min_disk}. Break!')
                break
            
            
            cluster_params = [[f'{cluster_idx}_{doc_idx}_{sent_idx}', sent, summary]
                for doc_idx, doc_sentences in enumerate(cluster_sentences)
                for sent_idx, sent in enumerate(doc_sentences)
            ]
            cluster_jsons = p.map(mp_core_doc, cluster_params)
            f.write('\n'.join(cluster_jsons)+'\n')
    p.close()
    p.join()
    os.rmdir(DP_TEMP)


# def gathered_mp_dump_sentence_cnndm(dataset_var, start=0, end=-1, min_disk=20):
#     if not exists(DP_TEMP):
#         os.mkdir(DP_TEMP)

#     if not exists(DP_SENTENCE_CNNDM):
#         os.mkdir(DP_SENTENCE_CNNDM)

#     if dataset_var not in DATASET_VARS:
#         raise ValueError(f'Illegal {dataset_var}!')

#     if end != -1 and end < start:
#         raise ValueError(f'Illegal end: {end}!')
    
#     # if start == 0:
#     #     dump_fp = DP_SENTENCE_CNNDM / f'{dataset_var}.json'
#     # else:
#     #     dump_fp = DP_SENTENCE_CNNDM / f'{dataset_var}_{start}.json'

#     if end != -1:
#         total = end
#         dump_fp = DP_SENTENCE_CNNDM / f'{dataset_var}_{start}_{end}.json'
#     else:
#         total = CNNDM_DATASET_SIZE[dataset_var]
#         dump_fp = DP_SENTENCE_CNNDM / f'{dataset_var}_{start}_EOS.json'

#     if exists(dump_fp):
#         raise ValueError(f'Remove {dump_fp} before dumping data')
    
#     logger.info(f'Ready to dump json data to {dump_fp}')

#     parser = CnnDmParser(dataset_var=dataset_var)
#     sample_generator = parser.sample_generator()

#     p = Pool(N_PROC)
#     with open(dump_fp, 'a') as f:
#         for doc_idx, doc_sentences, summary in tqdm(sample_generator, total=total):
#             if doc_idx < start:
#                 continue

#             if end != -1 and doc_idx >= end:
#                 break

#             if doc_idx % 50 == 0 and not check_disk(min_disk=min_disk):
#                 print(f'Disk less than {min_disk}. Break!')
#                 break

#             params = [[f'{doc_idx}_{sent_idx}', sent, summary]
#                 for sent_idx, sent in enumerate(doc_sentences)
#             ]
#             jsons = p.map(mp_core_doc, params)
#             f.write('\n'.join(jsons)+'\n')
#     p.close()
#     p.join()
#     os.rmdir(DP_TEMP)


if __name__ == "__main__":
    dataset_var = 'train'
    start = 0
    end = -1
    min_disk = 5
    gathered_mp_dump_sentence_hindi(dataset_var=dataset_var, start=start, min_disk=min_disk)

    # gathered_mp_dump_sentence_mn(dataset_var=dataset_var, start=start, min_disk=min_disk)
    # gathered_mp_dump_sentence_cnndm(dataset_var=dataset_var, start=start, end=end, min_disk=min_disk)
