#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt


def load_json_list(path: Path) -> Optional[list]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else None
    except Exception:
        return None


def load_metric_map(
    algo_dir: Path,
    dataset: str,
    filename: str,
    value_keys: Tuple[str, ...],
) -> Dict[int, float]:
    """
    Load a metric from algo_dir/dataset/filename.
    Returns {k: value}.
    value_keys: possible keys for the value in each item (e.g. ('runtime','time')).
    """
    dataset_dir = algo_dir / dataset
    data = load_json_list(dataset_dir / filename)
    if not data:
        return {}

    out: Dict[int, float] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        k = item.get("k")
        if k is None:
            continue

        val = None
        for key in value_keys:
            if key in item and item[key] is not None:
                val = item[key]
                break

        if val is None:
            continue

        try:
            out[int(k)] = float(val)
        except Exception:
            continue

    return out


def plot_runtime(dataset: str, outdir: Path, ce: dict, ex: dict) -> None:
    ks = sorted(set(ce.keys()) | set(ex.keys()))
    if not ks:
        return

    x_ce = [k for k in ks if k in ce]
    y_ce = [ce[k] for k in x_ce]

    x_ex = [k for k in ks if k in ex]
    y_ex = [ex[k] for k in x_ex]

    plt.figure(figsize=(8, 6))
    if x_ce:
        plt.plot(
            x_ce,
            y_ce,
            color="red",
            linestyle="-",
            linewidth=2,
            marker="o",
            label="TKUS-CE",
        )
    if x_ex:
        plt.plot(
            x_ex,
            y_ex,
            color="black",
            linestyle="--",
            linewidth=2,
            marker="*",
            label="TKHUSP-Miner",
        )

    plt.title(dataset, fontsize=18)
    plt.xlabel("k", fontsize=14)
    plt.ylabel("Time (s)", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()

    outfile = outdir / f"runtime_{dataset}.pdf"
    plt.savefig(outfile, format="pdf")
    plt.close()


def plot_memory(dataset: str, outdir: Path, ce: dict, ex: dict) -> None:
    ks = sorted(set(ce.keys()) | set(ex.keys()))
    if not ks:
        return

    x_ce = [k for k in ks if k in ce]
    y_ce = [ce[k] for k in x_ce]

    x_ex = [k for k in ks if k in ex]
    y_ex = [ex[k] for k in x_ex]

    plt.figure(figsize=(8, 6))
    if x_ce:
        plt.plot(
            x_ce,
            y_ce,
            color="red",
            linestyle="-",
            linewidth=2,
            marker="o",
            label="TKUS-CE",
        )
    if x_ex:
        plt.plot(
            x_ex,
            y_ex,
            color="black",
            linestyle="--",
            linewidth=2,
            marker="*",
            label="TKHUSP-Miner",
        )

    plt.title(dataset, fontsize=18)
    plt.xlabel("k", fontsize=14)
    plt.ylabel("Memory (MB)", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()

    outfile = outdir / f"mem_{dataset}.pdf"
    plt.savefig(outfile, format="pdf")
    plt.close()


def plot_acc_and_ar(
    dataset: str,
    outdir: Path,
    acc_ce: Dict[int, float],
    acc_ex: Dict[int, float],
    util_ce: Dict[int, float],
    util_ex: Dict[int, float],
) -> None:
    ks = sorted(
        set(acc_ce.keys())
        | set(acc_ex.keys())
        | set(util_ce.keys())
        | set(util_ex.keys())
    )
    if not ks:
        return

    # ---- Accuracy% ----
    # Exact: 100%
    # CE: (acc_CE / acc_Exact)*100 ; fallback acc_Exact = k if missing
    acc_ce_pct: Dict[int, float] = {}
    for k in ks:
        denom = acc_ex.get(k)
        if denom is None or denom <= 0:
            denom = float(k)
        num = acc_ce.get(k)
        if num is not None and denom > 0:
            acc_ce_pct[k] = (num / denom) * 100.0

    # ---- AR% ----
    # CE: (util_CE / util_Exact)*100
    ar_ce_pct: Dict[int, float] = {}
    for k in ks:
        u_ex = util_ex.get(k)
        u_ce = util_ce.get(k)
        if u_ex is not None and u_ex != 0 and u_ce is not None:
            ar_ce_pct[k] = (u_ce / u_ex) * 100.0

    x_acc = sorted(acc_ce_pct.keys())
    y_acc = [acc_ce_pct[k] for k in x_acc]

    x_ar = sorted(ar_ce_pct.keys())
    y_ar = [ar_ce_pct[k] for k in x_ar]

    # Baseline exact line (single legend entry)
    ks_baseline = sorted(set(x_acc) | set(x_ar))
    if not ks_baseline:
        return
    y_base = [100.0 for _ in ks_baseline]

    plt.figure(figsize=(9, 6))

    # TKHUSP-Miner baseline (single curve)
    plt.plot(
        ks_baseline,
        y_base,
        color="black",
        linestyle="--",
        linewidth=2,
        marker="*",
        label="TKHUSP-Miner",
    )

    # TKUS-CE Accuracy (red squares)
    if x_acc:
        plt.plot(
            x_acc,
            y_acc,
            color="red",
            linestyle="-",
            linewidth=2,
            marker="s",
            label="TKUS-CE Accuracy",
        )

    # TKUS-CE AR (blue plus)
    if x_ar:
        plt.plot(
            x_ar,
            y_ar,
            color="blue",
            linestyle="-",
            linewidth=2,
            marker="o",
            label="TKUS-CE AR",
        )

    plt.title(dataset, fontsize=18)
    plt.xlabel("k", fontsize=14)
    plt.ylabel("Percentage (%)", fontsize=14)
    plt.ylim(0, 105)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()

    outfile = outdir / f"acc_{dataset}.pdf"
    plt.savefig(outfile, format="pdf")
    plt.close()


def main(base_dir: str = ".", out_dir: str = "plots_pdf") -> None:
    base = Path(base_dir)
    dir_ce = base / "TKUS-CE"
    dir_ex = base / "TKHUSP-Miner"

    outdir = base / out_dir
    outdir.mkdir(parents=True, exist_ok=True)

    # Datasets = union of subdirs
    datasets = set()
    if dir_ce.exists():
        datasets |= {p.name for p in dir_ce.iterdir() if p.is_dir()}
    if dir_ex.exists():
        datasets |= {p.name for p in dir_ex.iterdir() if p.is_dir()}

    for dataset in sorted(datasets):
        # runtime
        rt_ce = load_metric_map(dir_ce, dataset, "runtime.json", ("runtime", "time"))
        rt_ex = load_metric_map(dir_ex, dataset, "runtime.json", ("runtime", "time"))
        plot_runtime(dataset, outdir, rt_ce, rt_ex)

        # memory
        mem_ce = load_metric_map(dir_ce, dataset, "memory.json", ("memory",))
        mem_ex = load_metric_map(dir_ex, dataset, "memory.json", ("memory",))
        plot_memory(dataset, outdir, mem_ce, mem_ex)

        # accuracy + AR
        acc_ce = load_metric_map(dir_ce, dataset, "acc.json", ("accuracy", "acc"))
        acc_ex = load_metric_map(dir_ex, dataset, "acc.json", ("accuracy", "acc"))

        util_ce = load_metric_map(dir_ce, dataset, "avgUtil.json", ("avgUtility", "avgUtil"))
        util_ex = load_metric_map(dir_ex, dataset, "avgUtil.json", ("avgUtility", "avgUtil"))

        plot_acc_and_ar(dataset, outdir, acc_ce, acc_ex, util_ce, util_ex)

    print(f"Done. PDFs saved in: {outdir.resolve()}")


if __name__ == "__main__":
    main()