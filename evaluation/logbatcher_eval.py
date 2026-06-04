"""
This file is part of TA-Eval-Rep.
Copyright (C) 2022 University of Luxembourg
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 3 of the License.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import sys
import os

sys.path.append('../')

from evaluation.settings import benchmark_settings
from evaluation.utils.common import common_args
from evaluation.utils.evaluator_main import evaluator, prepare_results
from evaluation.utils.postprocess import post_average

def should_skip_dataset(result_file, dataset):
    """Check if the dataset should be skipped based on the result file."""
    if not os.path.exists(result_file):
        return False
    with open(result_file, 'r') as file:
        lines = file.readlines()
    flag = False
    for line in lines:
        if line.startswith(dataset):
            parts = line.strip().split(',')
            if all(part not in ('', 'None') for part in parts[1:]):
                flag = True
            else:
                lines.remove(line)
    
    with open(result_file, 'w') as file:
        file.writelines(lines)
    return flag

datasets = [
    "Proxifier",
    "Linux",
    "Apache",
    "Zookeeper",
    "Hadoop",
    "HealthApp",
    "OpenStack",
    "HPC",
    "Mac",
    "OpenSSH",
    "Spark",
    "Thunderbird",
    "BGL",
    "HDFS"
]

if __name__ == "__main__":
    args = common_args()
    input_dir = "../datasets/sample10k_dataset/"
    output_dir = f"../outputs/parser/{args.config}" 

    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory {output_dir} does not exist.")
    

    # prepare results file
    result_file = prepare_results(
        output_dir=output_dir
    )
    if args.dataset != "null":
        datasets = [args.dataset]



    for dataset in datasets:
        if should_skip_dataset(os.path.join(output_dir, result_file), dataset):
            print(f"Skipping dataset {dataset} as it already has valid results.")
            continue

        setting = benchmark_settings[dataset]
        log_file = setting['log_file'].replace("_2k", f"_{args.data_type}")
        if os.path.exists(os.path.join(output_dir, f"{dataset}.log_structured.csv")):
            raise FileExistsError(f"parsing result of dataset {dataset} not exist.")
        
        # run evaluator for a dataset
        # The file is only for evalutation, so we remove the parameter "LogParser"
        evaluator(
            dataset=dataset,
            input_dir=input_dir,
            output_dir=output_dir,
            log_file=log_file,
            result_file=result_file,
        )  # it internally saves the results into a summary file
    metric_file = os.path.join(output_dir, result_file)
    
    if args.dataset == "null":
        post_average(metric_file)
    
