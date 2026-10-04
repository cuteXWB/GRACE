import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    return parser.parse_args()


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def harmonic_mean(a, b):
    return 2.0 * a * b / (a + b) if a + b > 0 else 0.0


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    root = Path(args.root)
    rows = []
    for summary in root.glob("*/summary_*_GRACE_id_ood.csv"):
        with summary.open(newline="", encoding="utf-8-sig") as stream:
            rows.extend(csv.DictReader(stream))
    if not rows:
        raise FileNotFoundError(f"No GRACE summary CSV files under {root}")

    id_efficiency = []
    for checkpoint in ("best", "latest"):
        for percentage in (10, 25, 50, 100):
            selected = [
                row
                for row in rows
                if row["checkpoint"] == checkpoint
                and row["split"] == "id"
                and int(row["train_percentage"]) == percentage
            ]
            dsc = mean([float(row["mean_dsc"]) for row in selected])
            nsd = mean([float(row["mean_nsd"]) for row in selected])
            id_efficiency.append(
                {
                    "checkpoint": checkpoint,
                    "train_percentage": percentage,
                    "num_source_datasets": len(selected),
                    "mean_id_dsc": f"{dsc:.8f}",
                    "mean_id_nsd": f"{nsd:.8f}",
                    "mean_id_dsc_percent": f"{dsc * 100:.4f}",
                    "mean_id_nsd_percent": f"{nsd * 100:.4f}",
                }
            )

    domain_generalization = []
    for checkpoint in ("best", "latest"):
        ood_rows = [
            row
            for row in rows
            if row["checkpoint"] == checkpoint and row["split"] == "ood"
        ]
        source_names = {row["source_dataset"] for row in ood_rows}
        id_rows = [
            row
            for row in rows
            if row["checkpoint"] == checkpoint
            and row["split"] == "id"
            and int(row["train_percentage"]) == 100
            and row["source_dataset"] in source_names
        ]
        id_dsc = mean([float(row["mean_dsc"]) for row in id_rows])
        id_nsd = mean([float(row["mean_nsd"]) for row in id_rows])
        ood_dsc = mean([float(row["mean_dsc"]) for row in ood_rows])
        ood_nsd = mean([float(row["mean_nsd"]) for row in ood_rows])
        domain_generalization.append(
            {
                "checkpoint": checkpoint,
                "num_id_sources": len(id_rows),
                "num_ood_targets": len(ood_rows),
                "id_dsc_percent": f"{id_dsc * 100:.4f}",
                "ood_dsc_percent": f"{ood_dsc * 100:.4f}",
                "hm_dsc_percent": f"{harmonic_mean(id_dsc, ood_dsc) * 100:.4f}",
                "id_nsd_percent": f"{id_nsd * 100:.4f}",
                "ood_nsd_percent": f"{ood_nsd * 100:.4f}",
                "hm_nsd_percent": f"{harmonic_mean(id_nsd, ood_nsd) * 100:.4f}",
            }
        )

    write_csv(root / "summary_GRACE_id_data_efficiency.csv", id_efficiency)
    write_csv(
        root / "summary_GRACE_domain_generalization.csv",
        domain_generalization,
    )
    print(f"Wrote aggregate summaries under {root}")


if __name__ == "__main__":
    main()
