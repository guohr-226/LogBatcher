import csv
import json
import os
import sys
import time
import pandas as pd
from collections import Counter
from tqdm import tqdm
from logbatcher.vars import vars_update
from logbatcher.cluster import Cluster,tokenize, vectorize, cluster, reassign_clusters, process_new_cluster
from logbatcher.additional_cluster import hierichical_clustering,meanshift_clustering
from logbatcher.util import verify_template
from logbatcher.parsing_cache import ParsingCache

USE_PROGRESS_BAR = sys.stdout.isatty() and sys.stderr.isatty()

def _elapsed(start_time):
    return f"{time.time() - start_time:.2f}s"

def _progress(message):
    if USE_PROGRESS_BAR:
        tqdm.write(message)
    else:
        print(message, flush=True)

def single_dataset_paring(dataset, contents, output_dir, parser, batch_size = 10, chunk_size = 10000, clustering_method = 'dbscan', debug=True):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logs = contents
    log_chunk = []
    log_chunk_index = []
    caching = ParsingCache()
    print(f'Parsing {len(logs)} logs in dataset {dataset}...')

    outputs = [None for _ in range(len(logs))]
    outputs_index = [None for _ in range(len(logs))]
    cache_matched_logs = 0
    
    # Parsing
    t1 = time.time()
    chunk_id = 0
    iterable = tqdm(
        enumerate(logs),
        total=len(logs),
        unit="log",
        desc=f"{dataset} logs",
        disable=not USE_PROGRESS_BAR,
        dynamic_ncols=USE_PROGRESS_BAR
    )
    for index, log in iterable:

        match_results = caching.match_event(log)
        if match_results[0] != "NoMatch":
            cache_matched_logs += 1
            # outputs[index] = match_results[0]
            outputs_index[index] = match_results[1]
        else:
            log_chunk.append(log)
            log_chunk_index.append(index)
        

        # Parsing with LLM
        if len(log_chunk) == chunk_size or (len(log_chunk)!=0 and index == len(logs) - 1):
            # parsing start
            chunk_id += 1
            chunk_start = time.time()
            chunk_range = f"{log_chunk_index[0]}-{log_chunk_index[-1]}"
            if debug:
                _progress(
                    f'[{dataset}] chunk {chunk_id} start: '
                    f'logs={len(log_chunk)}, source_index={chunk_range}, '
                    f'cache_hits={caching.hit_num}, templates={len(set(caching.template_list))}'
                )
            if clustering_method == 'dbscan':
                # tokenize -> vectorize -> cluster -> reassign_clusters
                cluster_stage_start = time.time()
                _progress(f'[{dataset}] chunk {chunk_id}: tokenize/vectorize/dbscan start') if debug else None
                tokenized_logs = [tokenize(log) for log in log_chunk]
                labels, cluster_nums = cluster(vectorize(tokenized_logs))
                labels, cluster_nums = reassign_clusters(labels, cluster_nums, tokenized_logs)
                _progress(
                    f'[{dataset}] chunk {chunk_id}: dbscan done in {_elapsed(cluster_stage_start)}, '
                    f'clusters={cluster_nums}'
                ) if debug else None
            elif clustering_method == 'hierarchical':
                cluster_stage_start = time.time()
                _progress(f'[{dataset}] chunk {chunk_id}: hierarchical clustering start') if debug else None
                labels, cluster_nums = hierichical_clustering(log_chunk)
                _progress(
                    f'[{dataset}] chunk {chunk_id}: hierarchical clustering done in {_elapsed(cluster_stage_start)}, '
                    f'clusters={cluster_nums}'
                ) if debug else None
            elif clustering_method == 'meanshift':
                cluster_stage_start = time.time()
                _progress(f'[{dataset}] chunk {chunk_id}: meanshift clustering start') if debug else None
                labels, cluster_nums = meanshift_clustering(log_chunk)
                _progress(
                    f'[{dataset}] chunk {chunk_id}: meanshift clustering done in {_elapsed(cluster_stage_start)}, '
                    f'clusters={cluster_nums}'
                ) if debug else None
            else:
                raise ValueError('Invalid clustering method')

            # create clusters
            build_stage_start = time.time()
            clusters = [None for _ in range(cluster_nums)]
            for index, label in enumerate(labels):
                if clusters[label] is None:
                    clusters[label] = Cluster()
                clusters[label].append_log(log_chunk[index], log_chunk_index[index])

            # sorting
            clusters = sorted(clusters, key=lambda cluster: len(cluster.logs), reverse=True)

            # batching
            [cluster.batching(batch_size) for cluster in clusters]
            top_sizes = [len(cluster.logs) for cluster in clusters[:5]]
            _progress(
                f'[{dataset}] chunk {chunk_id}: cluster objects ready in {_elapsed(build_stage_start)}, '
                f'top_sizes={top_sizes}'
            ) if debug else None

            # parsing
            # print(len(clusters), 'clusters identified') if debug else None  
            cluster_index = 0
            cluster_bar = tqdm(
                total=len(clusters),
                unit="cluster",
                desc=f"{dataset} chunk {chunk_id} LLM",
                leave=False,
                disable=not USE_PROGRESS_BAR,
                dynamic_ncols=USE_PROGRESS_BAR
            )
            while cluster_index < len(clusters):
                if cluster_bar.total != len(clusters):
                    cluster_bar.total = len(clusters)
                    cluster_bar.refresh()
                old_cluster = clusters[cluster_index]
                if debug:
                    _progress(
                        f'[{dataset}] chunk {chunk_id} cluster {cluster_index + 1}/{len(clusters)} '
                        f'start: size={old_cluster.size}, samples={len(old_cluster.batch_logs)}, '
                        f'llm_calls={parser.token_list[0]}, templates={len(set(caching.template_list))}'
                    )
                llm_start = time.time()
                template, old_cluster, new_cluster = parser.get_responce(old_cluster, cache_base = caching)
                if debug:
                    _progress(
                        f'[{dataset}] chunk {chunk_id} cluster {cluster_index + 1} done in {_elapsed(llm_start)}: '
                        f'template={template}'
                    )
                # update clusters
                cluster_nums += process_new_cluster(new_cluster, clusters, batch_size)
                refer_log = old_cluster.logs[0]
                if template not in caching.template_list:
                    if verify_template(template):
                        if debug:
                            print('=' * 20)
                            print(f'New cluster processed, {len(set(caching.template_list))} templates identified till now:')
                            print(f'Refer Log: {refer_log}')
                            print(f'Output Template: {template}')
                        id, _, _ = caching.add_templates(event_template=template, insert=False, refer_log = refer_log)
                        caching.variable_candidates.extend(vars_update(refer_log, template, caching.variable_candidates))
                    else:
                        id, _, _ = caching.add_templates(event_template=refer_log, insert=False, refer_log = refer_log)
                else:
                    id = caching.template_list.index(template)
                for index in old_cluster.indexs:
                    outputs_index[index] = id
                cluster_index += 1
                cluster_bar.update(1)
            cluster_bar.close()
            _progress(
                f'[{dataset}] chunk {chunk_id} done in {_elapsed(chunk_start)}: '
                f'total_clusters={len(clusters)}, templates={len(set(caching.template_list))}, '
                f'llm_calls={parser.token_list[0]}'
            ) if debug else None
            log_chunk = []
            log_chunk_index = []
    
    print(caching.variable_candidates)
    outputs = [caching.template_list[i] for i in outputs_index]
    # Result
    t2 = time.time()
    print(f'parsing time: {t2 - t1}')
    print(f'idetified templates: {len(set(outputs))}')

    # output logs
    output_log_file = output_dir + f'{dataset}_full.log_structured.csv'
    df = pd.DataFrame({'Content': logs, 'EventTemplate': outputs})
    df.to_csv(output_log_file, index=False)

    # output templates
    counter = Counter(outputs)
    items = list(counter.items())
    items.sort(key=lambda x: x[1], reverse=True)
    output_template_file = output_dir + f'{dataset}_full.template_structured.csv'
    template_df = pd.DataFrame(items, columns=['EventTemplate', 'Occurrence'])
    template_df['EventID'] = [f"E{i + 1}" for i in range(len(template_df))]
    template_df[['EventID', 'EventTemplate', 'Occurrence']].to_csv(output_template_file, index=False)

    # Save time cost
    time_cost_file = output_dir + 'time_cost.json'
    time_table = {}
    if os.path.exists(time_cost_file):
        with open(time_cost_file, 'r') as file:
            time_table = json.load(file)
    time_table[dataset] = {
        'InvocatingTime': parser.time_consumption_llm.__round__(3),
        'ParsingTime': (t2 - t1).__round__(3),
        'HitNum': caching.hit_num,
        'CacheMatchedLogs': cache_matched_logs,
        'len_of_hashing_table': len(caching.hashing_cache),
        'TokenCount': parser.token_list,
        'LLMUsage': parser.get_llm_usage_metrics() if hasattr(parser, "get_llm_usage_metrics") else {},
        'R2RTraceMetrics': parser.get_r2r_trace_metrics() if hasattr(parser, "get_r2r_trace_metrics") else {},
        'TemplateRecords': caching.template_records,
    }
    with open(time_cost_file, 'w') as file:
        json.dump(time_table, file)

    llm_usage = parser.get_llm_usage_metrics() if hasattr(parser, "get_llm_usage_metrics") else {
        "invocations": parser.token_list[0],
        "prompt_tokens": parser.token_list[1],
        "completion_tokens": 0,
        "total_tokens": parser.token_list[1],
        "latency_sec": round(parser.time_consumption_llm, 3),
        "avg_latency_sec": (
            round(parser.time_consumption_llm / parser.token_list[0], 6)
            if parser.token_list[0] else 0
        ),
    }
    r2r_trace_metrics = (
        parser.get_r2r_trace_metrics()
        if hasattr(parser, "get_r2r_trace_metrics")
        else {
            "router_trigger_count": 0,
            "routed_token_count": 0,
            "token_trace_count": 0,
        }
    )

    r2r_metrics_file = output_dir + 'r2r_metrics.json'
    r2r_metrics = {}
    if os.path.exists(r2r_metrics_file):
        with open(r2r_metrics_file, 'r') as file:
            r2r_metrics = json.load(file)

    parsing_time = round(t2 - t1, 3)
    total_logs = len(logs)
    r2r_metrics[dataset] = {
        "dataset": dataset,
        "total_logs": total_logs,
        "cache_matched_logs": cache_matched_logs,
        "hash_cache_hits": caching.hit_num,
        "cache_hit_rate": cache_matched_logs / total_logs if total_logs else 0,
        "hash_cache_hit_rate": caching.hit_num / total_logs if total_logs else 0,
        "llm_invocations": llm_usage["invocations"],
        "llm_invocation_rate": llm_usage["invocations"] / total_logs if total_logs else 0,
        "llm_prompt_tokens": llm_usage["prompt_tokens"],
        "llm_completion_tokens": llm_usage["completion_tokens"],
        "llm_total_tokens": llm_usage["total_tokens"],
        "llm_tokens_per_log": llm_usage["total_tokens"] / total_logs if total_logs else 0,
        "llm_latency_sec": llm_usage["latency_sec"],
        "llm_avg_latency_sec": llm_usage["avg_latency_sec"],
        "parsing_time_sec": parsing_time,
        "latency_per_log_sec": (t2 - t1) / total_logs if total_logs else 0,
        "router_trigger_count": r2r_trace_metrics["router_trigger_count"],
        "routed_token_count": r2r_trace_metrics["routed_token_count"],
        "token_trace_count": r2r_trace_metrics["token_trace_count"],
        "template_records": caching.template_records,
    }
    with open(r2r_metrics_file, 'w') as file:
        json.dump(r2r_metrics, file, indent=2)

    r2r_metrics_csv = output_dir + 'r2r_metrics.csv'
    csv_fields = [
        "dataset",
        "total_logs",
        "cache_matched_logs",
        "hash_cache_hits",
        "cache_hit_rate",
        "hash_cache_hit_rate",
        "llm_invocations",
        "llm_invocation_rate",
        "llm_prompt_tokens",
        "llm_completion_tokens",
        "llm_total_tokens",
        "llm_tokens_per_log",
        "llm_latency_sec",
        "llm_avg_latency_sec",
        "parsing_time_sec",
        "latency_per_log_sec",
        "router_trigger_count",
        "routed_token_count",
        "token_trace_count",
    ]
    with open(r2r_metrics_csv, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()
        for dataset_name, metrics in sorted(r2r_metrics.items()):
            writer.writerow({
                "dataset": dataset_name,
                "total_logs": metrics["total_logs"],
                "cache_matched_logs": metrics["cache_matched_logs"],
                "hash_cache_hits": metrics["hash_cache_hits"],
                "cache_hit_rate": metrics["cache_hit_rate"],
                "hash_cache_hit_rate": metrics["hash_cache_hit_rate"],
                "llm_invocations": metrics["llm_invocations"],
                "llm_invocation_rate": metrics["llm_invocation_rate"],
                "llm_prompt_tokens": metrics["llm_prompt_tokens"],
                "llm_completion_tokens": metrics["llm_completion_tokens"],
                "llm_total_tokens": metrics["llm_total_tokens"],
                "llm_tokens_per_log": metrics["llm_tokens_per_log"],
                "llm_latency_sec": metrics["llm_latency_sec"],
                "llm_avg_latency_sec": metrics["llm_avg_latency_sec"],
                "parsing_time_sec": metrics["parsing_time_sec"],
                "latency_per_log_sec": metrics["latency_per_log_sec"],
                "router_trigger_count": metrics["router_trigger_count"],
                "routed_token_count": metrics["routed_token_count"],
                "token_trace_count": metrics["token_trace_count"],
            })
