import argparse
import json
import os
import sys
import pandas as pd
from tqdm import tqdm
from logbatcher.parser import Parser
from logbatcher.util import generate_logformat_regex, log_to_dataframe
from logbatcher.parsing_base import single_dataset_paring

USE_PROGRESS_BAR = sys.stdout.isatty() and sys.stderr.isatty()

def progress_write(message):
    if USE_PROGRESS_BAR:
        tqdm.write(message)
    else:
        print(message, flush=True)

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='the Large Lauguage model used in LogBatcher, default to be gpt-4o-mini.')
    parser.add_argument('--batch_size', type=int, default=10, 
                        help='The size of a batch.')
    parser.add_argument('--chunk_size', type=int, default=10000,
                        help='Size of logs in a chunk.')
    parser.add_argument('--config', type=str, default="null")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()

    datasets = ['BGL', 'HDFS', 'OpenStack', 'OpenSSH', 'HPC', 'Zookeeper', 'Spark', 'Proxifier', 'HealthApp', 'Mac', 'Hadoop', 'Apache', 'Linux', 'Thunderbird']

    # output dir
    if args.config == 'null':
        output_folder = f"logbatcher"
    else:
        output_folder = args.config
    output_dir = f'outputs/parser/{output_folder}/'

    # load api key and dataset format
    with open('config.json', 'r') as f:
        config = json.load(f)
    parser = Parser(args.model, output_folder, config)
    print(f"Benchmark output directory: {output_dir}", flush=True)
    dataset_iter = tqdm(
        datasets,
        desc="Datasets",
        unit="dataset",
        disable=not USE_PROGRESS_BAR,
        dynamic_ncols=USE_PROGRESS_BAR
    )
    for index, dataset in enumerate(dataset_iter):
        if USE_PROGRESS_BAR:
            dataset_iter.set_postfix_str(dataset)
        if os.path.exists(f'{output_dir}{dataset}_full.log_structured.csv'):
            progress_write(f'{dataset} has been parsed, skip it.')
            continue
        progress_write(f'[{index + 1}/{len(datasets)}] Start dataset {dataset}')
        structured_log_file = f'datasets/sample10k_dataset/{dataset}/{dataset}_full.log_structured.csv'
        
        log_file_format = 'structured'
        if log_file_format == 'structured':
            df = pd.read_csv(structured_log_file)
            logs = df['Content'].tolist()
        elif log_file_format == 'raw':
            log_file = f'dataset/{dataset}/{dataset}.log'
            with open(log_file, 'r') as f:
                log_raws = f.readlines()
            headers, regex = generate_logformat_regex(config['datasets_format'][dataset])
            logs = log_to_dataframe(log_file, regex, headers, len(log_raws))
        else:
            raise ValueError('log_file_format should be structured or raw')
        
        parser.dataset = dataset
        single_dataset_paring(
            dataset=dataset,
            contents=logs,
            output_dir=output_dir, 
            parser=parser, 
            batch_size=args.batch_size,
            chunk_size=args.chunk_size
        )
        print('time cost by llm: ', parser.time_consumption_llm)
        parser.time_consumption_llm = 0
        parser.token_list = [0, 0]
