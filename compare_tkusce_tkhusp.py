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

    def parse_runtime_exact(self) -> Dict[str, Dict[int, Dict[str, float]]]:
        """
        Parses runtime_exacte.txt to get TKHUSP-Miner runtimes and memory usage.
        Returns: {dataset: {k: {'runtime': float, 'memory': float}}}
        """
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
            
            # Find all k and their runtimes/memory
            parts = re.split(r'✅\s*k\s*=\s*(\d+)', block_content)
            for j in range(1, len(parts), 2):
                try:
                    k = int(parts[j])
                    k_block = parts[j + 1]
                    
                    metrics = {}
                    
                    # Runtime
                    match_time = re.search(r'Execution time\s*=\s*([\d\.]+)\s*s', k_block)
                    if match_time:
                        metrics['runtime'] = float(match_time.group(1))
                        
                    # Memory
                    match_mem = re.search(r'Max memory\s*=\s*([\d\.]+)\s*MB', k_block)
                    if match_mem:
                        metrics['memory'] = float(match_mem.group(1))
                        
                    if metrics:
                        result[dataset][k] = metrics
                        
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
        Loads acc.json, runtime.json, avgUtil.json, memory.json for TKUS-CE.
        Returns: {k: {'accuracy': val, 'runtime': val, 'avgUtil': val, 'memory': val}}
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
                                elif key_map == 'memory':
                                    val = item.get('memory')
                                    
                                if val is not None:
                                    data[k][key_map] = float(val)
            except Exception as e:
                print(f"Error loading {fpath}: {e}")

        load_file('acc.json', 'accuracy')
        load_file('runtime.json', 'runtime')
        load_file('avgUtil.json', 'avgUtil')
        load_file('memory.json', 'memory')
        
        return data

def generate_plots():
    parser = DataParser()
    
    # 1. Load Exact Runtimes and Memory
    exact_metrics_map = parser.parse_runtime_exact() # {dataset: {k: {runtime, memory}}}
    
    # 2. Identify all datasets
    datasets = set(exact_metrics_map.keys())
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
        exact_runtime_mem = exact_metrics_map.get(dataset, {})  # {k: {runtime, memory}}
        
        # Load TKUS-CE Data
        tkusce_data = parser.load_tkusce_data(dataset)  # {k: {accuracy, runtime, avgUtil, memory}}
        
        # Collect all K values
        all_ks = set(exact_results.keys())
        all_ks.update(exact_runtime_mem.keys())
        all_ks.update(tkusce_data.keys())
        
        if not all_ks:
            print(f"  No data for {dataset}, skipping.")
            continue
            
        sorted_ks = sorted(list(all_ks))
        
        # --- 1. Accuracy Plot ---
        fig_acc, ax_acc = plt.subplots(figsize=(8, 6))
        
        # TKHUSP-Miner is always 100%
        ax_acc.axhline(y=100, color=parser.colors['TKHUSP-Miner'], 
                       linestyle=parser.linestyles['TKHUSP-Miner'], 
                       label='TKHUSP-Miner', linewidth=2)
        
        # TKUS-CE
        xs = []
        ys = []
        for k in sorted_ks:
            if k in tkusce_data and 'accuracy' in tkusce_data[k]:
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
        
        plt.tight_layout()
        outfile_acc = output_dir / f"{dataset}_Accuracy.svg"
        plt.savefig(outfile_acc, format='svg')
        plt.close(fig_acc)
        print(f"  ✅ Saved {outfile_acc}")

        # --- 2. Runtime Plot ---
        fig_rt, ax_rt = plt.subplots(figsize=(8, 6))
        
        # TKHUSP-Miner
        ex_xs = []
        ex_ys = []
        for k in sorted_ks:
            if k in exact_runtime_mem and 'runtime' in exact_runtime_mem[k]:
                ex_xs.append(k)
                ex_ys.append(exact_runtime_mem[k]['runtime'])
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
        
        plt.tight_layout()
        outfile_rt = output_dir / f"{dataset}_Runtime.svg"
        plt.savefig(outfile_rt, format='svg')
        plt.close(fig_rt)
        print(f"  ✅ Saved {outfile_rt}")

        # --- 3. Average Utility Plot ---
        fig_avg, ax_avg = plt.subplots(figsize=(8, 6))
        
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
        
        plt.tight_layout()
        outfile_avg = output_dir / f"{dataset}_AvgUtility.svg"
        plt.savefig(outfile_avg, format='svg')
        plt.close(fig_avg)
        print(f"  ✅ Saved {outfile_avg}")

        # --- 4. Memory Plot (Histogram) ---
        fig_mem, ax_mem = plt.subplots(figsize=(10, 6))
        
        # Prepare data for histogram
        ks_for_mem = sorted_ks
        x = np.arange(len(ks_for_mem))
        width = 0.35
        
        exact_mem_vals = []
        tkusce_mem_vals = []
        
        for k in ks_for_mem:
            # Exact Memory
            val_ex = 0
            if k in exact_runtime_mem and 'memory' in exact_runtime_mem[k]:
                val_ex = exact_runtime_mem[k]['memory']
            exact_mem_vals.append(val_ex)
            
            # TKUS-CE Memory
            val_ce = 0
            if k in tkusce_data and 'memory' in tkusce_data[k]:
                val_ce = tkusce_data[k]['memory']
            tkusce_mem_vals.append(val_ce)
            
        # Draw bars
        # Only draw if there is data
        has_data = any(v > 0 for v in exact_mem_vals) or any(v > 0 for v in tkusce_mem_vals)
        
        if has_data:
            rects1 = ax_mem.bar(x - width/2, exact_mem_vals, width, label='TKHUSP-Miner', color=parser.colors['TKHUSP-Miner'])
            rects2 = ax_mem.bar(x + width/2, tkusce_mem_vals, width, label='TKUS-CE', color=parser.colors['TKUS-CE'])
            
            ax_mem.set_title(f"{dataset} - Memory Consumption", fontsize=14, fontweight='bold')
            ax_mem.set_xlabel("k", fontsize=12)
            ax_mem.set_ylabel("Memory (MB)", fontsize=12)
            ax_mem.set_xticks(x)
            ax_mem.set_xticklabels(ks_for_mem)
            ax_mem.legend(fontsize=11)
            ax_mem.grid(True, linestyle='--', alpha=0.5, axis='y')
            
            plt.tight_layout()
            outfile_mem = output_dir / f"{dataset}_Memory.svg"
            plt.savefig(outfile_mem, format='svg')
            print(f"  ✅ Saved {outfile_mem}")
        else:
             print(f"  ⚠️ No memory data for {dataset}, skipping memory plot.")
        
        plt.close(fig_mem)

if __name__ == "__main__":
    print("=" * 60)
    print("TKUS-CE vs TKHUSP-Miner Comparison Plot Generator")
    print("=" * 60)
    generate_plots()
    print("=" * 60)
    print("✅ All plots generated successfully!")
    print("=" * 60)
