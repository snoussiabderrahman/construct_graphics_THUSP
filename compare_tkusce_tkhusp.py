import json
import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np

class DataParser:
    """Parses data for TKUS-CE and TKHUSP-Miner algorithms."""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        
        # TKUS-CE data directory
        self.tkusce_dir = self.base_dir / "filesJSON"
        
        # TKHUSP-Miner exact algorithm data
        self.output_exact_dir = self.base_dir / "output_exacts"
        self.runtime_exact_file = self.base_dir / "runtime_exacte.txt"
        
        # Colors and markers for the two algorithms
        self.colors = {
            'TKUS-CE': '#E63946',      # Red
            'TKHUSP-Miner': '#000000'   # Black
        }
        
        self.markers = {
            'TKUS-CE': 'o',
            'TKHUSP-Miner': '*'
        }
        
        self.linestyles = {
            'TKUS-CE': '-',
            'TKHUSP-Miner': '--'
        }

    def parse_runtime_exact(self) -> Dict[str, Dict[int, float]]:
        """Parses runtime_exacte.txt to get TKHUSP-Miner runtimes."""
        result = {}
        if not self.runtime_exact_file.exists():
            print(f"⚠️  File {self.runtime_exact_file} not found")
            return result
        
        try:
            content = self.runtime_exact_file.read_text(encoding='utf-8')
        except Exception as e:
            print(f"⚠️  Error reading {self.runtime_exact_file}: {e}")
            return result
        
        # Split by datasets
        blocks = re.split(r'-{5,}\s*DB\s*:\s*(.+?)\s*-{5,}', content, flags=re.DOTALL)
        
        for i in range(1, len(blocks), 2):
            dataset = blocks[i].strip()
            block_content = blocks[i + 1]
            result[dataset] = {}
            
            # Find all k and their runtimes
            # Pattern: ✅ k = 10 ... Execution time = 2.818 s
            parts = re.split(r'✅\s*k\s*=\s*(\d+)', block_content)
            for j in range(1, len(parts), 2):
                try:
                    k = int(parts[j])
                    k_block = parts[j + 1]
                    match = re.search(r'Execution time\s*=\s*([\d\.]+)\s*s', k_block)
                    if match:
                        result[dataset][k] = float(match.group(1))
                except Exception:
                    continue
        return result

    def get_exact_results(self, dataset: str) -> Dict[int, Dict[str, float]]:
        """
        Parses output files to get Avg Utility and Pattern Count for TKHUSP-Miner.
        Returns: {k: {'avgUtil': float, 'count': int}}
        """
        results = {}
        dataset_dir = self.output_exact_dir / dataset
        if not dataset_dir.exists():
            return results
            
        # Look for files like BIBLE_10.txt
        for file in dataset_dir.glob(f"{dataset}_*.txt"):
            try:
                # Extract k from filename
                match = re.search(r'_(\d+)\.txt$', file.name)
                if not match:
                    continue
                k = int(match.group(1))
                
                content = file.read_text(encoding='utf-8')
                utilities = []
                for line in content.splitlines():
                    if '#UTIL:' in line:
                        u_match = re.search(r'#UTIL:\s*([\d\.]+)', line)
                        if u_match:
                            utilities.append(float(u_match.group(1)))
                
                if utilities:
                    avg_util = sum(utilities) / len(utilities)
                    results[k] = {'avgUtil': avg_util, 'count': len(utilities)}
                else:
                    results[k] = {'avgUtil': 0.0, 'count': 0}
                    
            except Exception as e:
                print(f"Error parsing {file}: {e}")
                continue
                
        return results

    def load_tkusce_data(self, dataset: str) -> Dict[int, Dict[str, float]]:
        """
        Loads acc.json, runtime.json, avgUtil.json for TKUS-CE.
        Returns: {k: {'accuracy': val, 'runtime': val, 'avgUtil': val}}
        """
        data = {}
        dataset_dir = self.tkusce_dir / dataset
        if not dataset_dir.exists():
            return data
            
        # Helper to load and merge
        def load_file(filename: str, key_map: str):
            fpath = dataset_dir / filename
            if not fpath.exists():
                return
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    if isinstance(json_data, list):
                        for item in json_data:
                            k = item.get('k')
                            if k is not None:
                                if k not in data:
                                    data[k] = {}
                                
                                val = None
                                if key_map == 'accuracy':
                                    val = item.get('accuracy')
                                elif key_map == 'runtime':
                                    val = item.get('time') or item.get('runtime')
                                elif key_map == 'avgUtil':
                                    val = item.get('avgUtility') or item.get('avgUtil')
                                    
                                if val is not None:
                                    data[k][key_map] = float(val)
            except Exception as e:
                print(f"Error loading {fpath}: {e}")

        load_file('acc.json', 'accuracy')
        load_file('runtime.json', 'runtime')
        load_file('avgUtil.json', 'avgUtil')
        
        return data

def generate_plots():
    parser = DataParser()
    
    # 1. Load Exact Runtimes
    exact_runtimes = parser.parse_runtime_exact()
    
    # 2. Identify all datasets
    datasets = set(exact_runtimes.keys())
    if parser.tkusce_dir.exists():
        for d in parser.tkusce_dir.iterdir():
            if d.is_dir():
                datasets.add(d.name)
    
    # Create output directory
    output_dir = Path("plots_comparison_CE")
    output_dir.mkdir(exist_ok=True)
    
    for dataset in sorted(datasets):
        print(f"Processing {dataset}...")
        
        # Load Exact Data (TKHUSP-Miner)
        exact_results = parser.get_exact_results(dataset)  # {k: {avgUtil, count}}
        exact_runtime_map = exact_runtimes.get(dataset, {})  # {k: runtime}
        
        # Load TKUS-CE Data
        tkusce_data = parser.load_tkusce_data(dataset)  # {k: {accuracy, runtime, avgUtil}}
        
        # Collect all K values
        all_ks = set(exact_results.keys())
        all_ks.update(exact_runtime_map.keys())
        all_ks.update(tkusce_data.keys())
        
        if not all_ks:
            print(f"  No data for {dataset}, skipping.")
            continue
            
        sorted_ks = sorted(list(all_ks))
        
        # Setup Plot
        fig, (ax_acc, ax_rt, ax_avg) = plt.subplots(1, 3, figsize=(24, 6))
        
        # --- 1. Accuracy Plot ---
        # TKHUSP-Miner is always 100% (exact algorithm)
        ax_acc.axhline(y=100, color=parser.colors['TKHUSP-Miner'], 
                       linestyle=parser.linestyles['TKHUSP-Miner'], 
                       label='TKHUSP-Miner', linewidth=2)
        
        # TKUS-CE accuracy
        xs = []
        ys = []
        for k in sorted_ks:
            if k in tkusce_data and 'accuracy' in tkusce_data[k]:
                # Calculate percentage
                exact_count = exact_results.get(k, {}).get('count', 0)
                if exact_count > 0:
                    acc_pct = (tkusce_data[k]['accuracy'] / exact_count) * 100
                    xs.append(k)
                    ys.append(acc_pct)
        
        if xs:
            ax_acc.plot(xs, ys, marker=parser.markers['TKUS-CE'], 
                       color=parser.colors['TKUS-CE'], 
                       linestyle=parser.linestyles['TKUS-CE'], 
                       label='TKUS-CE', linewidth=2)
        
        ax_acc.set_title(f"{dataset} - Accuracy", fontsize=14, fontweight='bold')
        ax_acc.set_xlabel("k", fontsize=12)
        ax_acc.set_ylabel("Accuracy (%)", fontsize=12)
        ax_acc.set_ylim(0, 105)
        ax_acc.grid(True, linestyle='--', alpha=0.5)
        ax_acc.legend(fontsize=11)
        
        # --- 2. Runtime Plot ---
        # TKHUSP-Miner
        ex_xs = []
        ex_ys = []
        for k in sorted_ks:
            if k in exact_runtime_map:
                ex_xs.append(k)
                ex_ys.append(exact_runtime_map[k])
        if ex_xs:
            ax_rt.plot(ex_xs, ex_ys, marker=parser.markers['TKHUSP-Miner'], 
                      color=parser.colors['TKHUSP-Miner'], 
                      linestyle=parser.linestyles['TKHUSP-Miner'], 
                      label='TKHUSP-Miner', linewidth=2)
        
        # TKUS-CE
        xs = []
        ys = []
        for k in sorted_ks:
            if k in tkusce_data and 'runtime' in tkusce_data[k]:
                xs.append(k)
                ys.append(tkusce_data[k]['runtime'])
        if xs:
            ax_rt.plot(xs, ys, marker=parser.markers['TKUS-CE'], 
                      color=parser.colors['TKUS-CE'], 
                      linestyle=parser.linestyles['TKUS-CE'], 
                      label='TKUS-CE', linewidth=2)

        ax_rt.set_title(f"{dataset} - Runtime", fontsize=14, fontweight='bold')
        ax_rt.set_xlabel("k", fontsize=12)
        ax_rt.set_ylabel("Time (s)", fontsize=12)
        ax_rt.grid(True, linestyle='--', alpha=0.5)
        ax_rt.legend(fontsize=11)

        # --- 3. Average Utility Plot ---
        # TKHUSP-Miner
        ex_xs = []
        ex_ys = []
        for k in sorted_ks:
            if k in exact_results:
                ex_xs.append(k)
                ex_ys.append(exact_results[k]['avgUtil'])
        if ex_xs:
            ax_avg.plot(ex_xs, ex_ys, marker=parser.markers['TKHUSP-Miner'], 
                       color=parser.colors['TKHUSP-Miner'], 
                       linestyle=parser.linestyles['TKHUSP-Miner'], 
                       label='TKHUSP-Miner', linewidth=2)

        # TKUS-CE
        xs = []
        ys = []
        for k in sorted_ks:
            if k in tkusce_data and 'avgUtil' in tkusce_data[k]:
                xs.append(k)
                ys.append(tkusce_data[k]['avgUtil'])
        if xs:
            ax_avg.plot(xs, ys, marker=parser.markers['TKUS-CE'], 
                       color=parser.colors['TKUS-CE'], 
                       linestyle=parser.linestyles['TKUS-CE'], 
                       label='TKUS-CE', linewidth=2)

        ax_avg.set_title(f"{dataset} - Average Utility", fontsize=14, fontweight='bold')
        ax_avg.set_xlabel("k", fontsize=12)
        ax_avg.set_ylabel("Utility", fontsize=12)
        ax_avg.grid(True, linestyle='--', alpha=0.5)
        ax_avg.legend(fontsize=11)
        
        # Save
        plt.tight_layout()
        outfile = output_dir / f"{dataset}_comparison.svg"
        plt.savefig(outfile, format='svg')
        plt.close(fig)
        print(f"  ✅ Saved {outfile}")

if __name__ == "__main__":
    print("=" * 60)
    print("TKUS-CE vs TKHUSP-Miner Comparison Plot Generator")
    print("=" * 60)
    generate_plots()
    print("=" * 60)
    print("✅ All plots generated successfully!")
    print("=" * 60)
