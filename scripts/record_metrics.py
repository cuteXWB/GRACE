import argparse
import csv
from pathlib import Path


KEY_COLUMNS = [
    "checkpoint",
    "split",
    "train_percentage",
    "source_dataset",
    "target_dataset",
]
FIELDNAMES = KEY_COLUMNS + [
    "mean_dsc",
    "mean_nsd",
    "mean_dsc_percent",
    "mean_nsd_percent",
    "metrics_csv",
    "checkpoint_path",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--checkpoint", choices=["best", "latest"], required=True)
    parser.add_argument("--split", choices=["id", "ood"], required=True)
    parser.add_argument("--train-percentage", type=int, required=True)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--target-dataset", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    return parser.parse_args()


def read_metric_means(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No metric rows in {path}")
    dsc = [float(row["DSC"]) for row in rows]
    nsd = [float(row["NSD"]) for row in rows]
    return sum(dsc) / len(dsc), sum(nsd) / len(nsd)


def main():
    args = parse_args()
    summary_path = Path(args.summary)
    metrics_path = Path(args.metrics)
    mean_dsc, mean_nsd = read_metric_means(metrics_path)
    new_row = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "train_percentage": str(args.train_percentage),
        "source_dataset": args.source_dataset,
        "target_dataset": args.target_dataset,
        "mean_dsc": f"{mean_dsc:.8f}",
        "mean_nsd": f"{mean_nsd:.8f}",
        "mean_dsc_percent": f"{mean_dsc * 100:.4f}",
        "mean_nsd_percent": f"{mean_nsd * 100:.4f}",
        "metrics_csv": str(metrics_path),
        "checkpoint_path": args.checkpoint_path,
    }

    existing = []
    if summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8-sig") as stream:
            existing = list(csv.DictReader(stream))
    key = tuple(new_row[column] for column in KEY_COLUMNS)
    existing = [
        row
        for row in existing
        if tuple(row.get(column, "") for column in KEY_COLUMNS) != key
    ]
    existing.append(new_row)
    existing.sort(
        key=lambda row: (
            row["checkpoint"],
            row["split"],
            int(row["train_percentage"]),
            row["target_dataset"],
        )
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(existing)
    temporary.replace(summary_path)
    print(
        f"Recorded {args.checkpoint}/{args.split}/{args.target_dataset}: "
        f"DSC={mean_dsc * 100:.2f}% NSD={mean_nsd * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
