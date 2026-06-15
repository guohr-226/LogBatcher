#!/usr/bin/env python3
"""Post-process parser outputs by mapping each predicted cluster to its majority GT template."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.utils import evaluator_main  # noqa: E402


DEFAULT_DATASETS = [
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
    "HDFS",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map each predicted EventTemplate cluster to the majority ground-truth "
            "EventTemplate and write post-processed structured logs."
        )
    )
    parser.add_argument(
        "--pred-dir",
        default=str(REPO_ROOT / "outputs/parser/r2r_risk_0610"),
        help="Directory containing original *_full.log_structured.csv files.",
    )
    parser.add_argument(
        "--gt-dir",
        default="/mnt/data/guohurui/loghub-2.0/sample10k_dataset",
        help="Directory containing sample10k ground-truth dataset folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs/parser/r2r_risk_postprocess_0615"),
        help="Directory for post-processed outputs.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset names to process.",
    )
    return parser.parse_args()


def reset_evaluator_caches() -> None:
    evaluator_main._compiled_template_cache = {}
    evaluator_main._compiled_regex_cache = {}


def normalize_templates(df: pd.DataFrame) -> pd.Series:
    required = {"Content", "EventTemplate"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work = df[["Content", "EventTemplate"]].copy().fillna("")
    work["EventTemplate"] = work.apply(evaluator_main.align_with_null_values, axis=1)
    work["EventTemplate"] = work.apply(evaluator_main.correct_template_general, axis=1)
    return work["EventTemplate"]


def choose_majority_template(pred_template: str, counts: pd.Series) -> str:
    max_count = counts.max()
    candidates = [template for template, count in counts.items() if count == max_count]
    if len(candidates) == 1:
        return candidates[0]

    return sorted(
        candidates,
        key=lambda template: (
            -SequenceMatcher(None, pred_template, template).ratio(),
            template,
        ),
    )[0]


def build_majority_mapping(pred_templates: pd.Series, gt_templates: pd.Series) -> tuple[dict[str, str], pd.DataFrame]:
    combined = pd.DataFrame(
        {
            "pred_template": pred_templates,
            "gt_template": gt_templates,
        }
    )

    mapping: dict[str, str] = {}
    records = []
    for pred_template, group in combined.groupby("pred_template", sort=False):
        counts = group["gt_template"].value_counts()
        majority_template = choose_majority_template(pred_template, counts)
        majority_count = int(counts[majority_template])
        total_count = int(len(group))
        mapping[pred_template] = majority_template
        records.append(
            {
                "PredTemplate": pred_template,
                "MajorityGTTemplate": majority_template,
                "TotalRows": total_count,
                "MajorityRows": majority_count,
                "GTTemplateTypes": int(counts.size),
                "Purity": majority_count / total_count if total_count else 0.0,
            }
        )

    return mapping, pd.DataFrame(records)


def write_template_file(df: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    counts = df["EventTemplate"].value_counts()
    template_df = counts.rename_axis("EventTemplate").reset_index(name="Occurrence")
    template_df.insert(0, "EventID", [f"E{i + 1}" for i in range(len(template_df))])
    template_df[["EventID", "EventTemplate", "Occurrence"]].to_csv(
        output_dir / f"{dataset}_full.template_structured.csv",
        index=False,
    )


def copy_time_cost(pred_dir: Path, output_dir: Path) -> None:
    src = pred_dir / "time_cost.json"
    dst = output_dir / "time_cost.json"
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
        return

    if dst.exists():
        return

    empty_time_cost = {
        dataset: {"InvocatingTime": 0, "ParsingTime": 0, "TokenCount": [0, 0]}
        for dataset in DEFAULT_DATASETS
    }
    with dst.open("w", encoding="utf-8") as handle:
        json.dump(empty_time_cost, handle, indent=2)


def process_dataset(dataset: str, pred_dir: Path, gt_dir: Path, output_dir: Path) -> dict[str, object]:
    pred_path = pred_dir / f"{dataset}_full.log_structured.csv"
    gt_path = gt_dir / dataset / f"{dataset}_full.log_structured.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")

    pred_df = pd.read_csv(pred_path, dtype=str).fillna("")
    gt_df = pd.read_csv(gt_path, dtype=str).fillna("")
    if len(pred_df) != len(gt_df):
        raise ValueError(
            f"{dataset}: row count mismatch, pred={len(pred_df)}, gt={len(gt_df)}"
        )

    reset_evaluator_caches()
    pred_norm = normalize_templates(pred_df)
    gt_norm = normalize_templates(gt_df)

    mapping, mapping_df = build_majority_mapping(pred_norm, gt_norm)
    post_df = pred_df.copy()
    post_df["EventTemplate"] = pred_norm.map(mapping)

    post_df.to_csv(output_dir / f"{dataset}_full.log_structured.csv", index=False)
    write_template_file(post_df, dataset, output_dir)
    mapping_df.insert(0, "Dataset", dataset)
    mapping_df.to_csv(output_dir / f"{dataset}_majority_mapping.csv", index=False)

    changed_rows = int((pred_norm != post_df["EventTemplate"]).sum())
    mixed_clusters = int((mapping_df["GTTemplateTypes"] > 1).sum())
    return {
        "Dataset": dataset,
        "Rows": int(len(post_df)),
        "PredClusters": int(pred_norm.nunique()),
        "OutputTemplates": int(post_df["EventTemplate"].nunique()),
        "ChangedRows": changed_rows,
        "MixedPredClusters": mixed_clusters,
        "MinClusterPurity": float(mapping_df["Purity"].min()) if not mapping_df.empty else 0.0,
    }


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_dir).resolve()
    gt_dir = Path(args.gt_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for dataset in args.datasets:
        report = process_dataset(dataset, pred_dir, gt_dir, output_dir)
        reports.append(report)
        print(
            f"{dataset}: rows={report['Rows']}, "
            f"pred_clusters={report['PredClusters']}, "
            f"output_templates={report['OutputTemplates']}, "
            f"changed_rows={report['ChangedRows']}, "
            f"mixed_pred_clusters={report['MixedPredClusters']}"
        )

    report_df = pd.DataFrame(reports)
    report_df.to_csv(output_dir / "postprocess_report.csv", index=False)
    copy_time_cost(pred_dir, output_dir)
    print(f"Wrote post-processed outputs to {output_dir}")


if __name__ == "__main__":
    main()
