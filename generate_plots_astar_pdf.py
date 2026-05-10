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
    Load a metric from algo_dir/dataset/filename for TKHUSP-Miner.
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

def load_astar_metric(algo_dir: Path, dataset: str, metric: str) -> Dict[int, float]:
    """
    Load a metric from algo_dir/metric/dataset.json for TKUS-Astar.
    """
    path = algo_dir / metric / f"{dataset}.json"
    if not path.exists():
        return {}
    
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            
        out = {}
        for k_str, val in data.items():
            try:
                if "_" in k_str:
                    k = int(k_str.split("_")[-1])
                else:
                    k = int(k_str)
                out[k] = float(val)
            except ValueError:
                continue
        return out
    except Exception:
        return {}


def plot_runtime(dataset: str, outdir: Path, astar: dict, ex: dict) -> None:
    ks = sorted(set(astar.keys()) | set(ex.keys()))
    if not ks:
        return

    x_astar = [k for k in ks if k in astar]
    y_astar = [astar[k] for k in x_astar]

    x_ex = [k for k in ks if k in ex]
    y_ex = [ex[k] for k in x_ex]

    plt.figure(figsize=(8, 6))
    if x_astar:
        plt.plot(
            x_astar,
            y_astar,
            color="red",
            linestyle="-",
            linewidth=2,
            marker="o",
            label="TKUS-Astar",
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

def plot_memory(dataset: str, outdir: Path, astar: dict, ex: dict) -> None:
    ks = sorted(set(astar.keys()) | set(ex.keys()))
    if not ks:
        return

    x_astar = [k for k in ks if k in astar]
    y_astar = [astar[k] for k in x_astar]

    x_ex = [k for k in ks if k in ex]
    y_ex = [ex[k] for k in x_ex]

    plt.figure(figsize=(8, 6))
    if x_astar:
        plt.plot(
            x_astar,
            y_astar,
            color="red",
            linestyle="-",
            linewidth=2,
            marker="o",
            label="TKUS-Astar",
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

def main(base_dir: str = ".", out_dir: str = "plots_astar_pdf") -> None:
    base = Path(base_dir)
    dir_astar = base / "TKUS-Astar"
    dir_ex = base / "TKHUSP-Miner"

    outdir = base / out_dir
    outdir.mkdir(parents=True, exist_ok=True)

    # Datasets from TKHUSP-Miner
    datasets = set()
    if dir_ex.exists():
        datasets |= {p.name for p in dir_ex.iterdir() if p.is_dir()}
        
    # Also datasets from TKUS-Astar/runtime
    if (dir_astar / "runtime").exists():
        datasets |= {p.stem for p in (dir_astar / "runtime").glob("*.json")}

    for dataset in sorted(datasets):
        # runtime
        rt_astar = load_astar_metric(dir_astar, dataset, "runtime")
        rt_ex = load_metric_map(dir_ex, dataset, "runtime.json", ("runtime", "time"))
        if rt_astar or rt_ex:
            plot_runtime(dataset, outdir, rt_astar, rt_ex)

        # memory
        mem_astar = load_astar_metric(dir_astar, dataset, "memory")
        mem_ex = load_metric_map(dir_ex, dataset, "memory.json", ("memory",))
        if mem_astar or mem_ex:
            plot_memory(dataset, outdir, mem_astar, mem_ex)

    print(f"Done. PDFs saved in: {outdir.resolve()}")

if __name__ == "__main__":
    main()
