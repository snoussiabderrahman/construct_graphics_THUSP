import json
import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np

class DataParser:
    """Parses data for HUSP-SP and TKUSP versions."""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        # The user mentioned 'output_husp', but the file system shows 'output_exacts'.
        # We will check for both, prioritizing 'output_husp' if it exists, else 'output_exacts'.
        if (self.base_dir / "output_husp").exists():
             self.output_exact_dir = self.base_dir / "output_husp"
        else:
             self.output_exact_dir = self.base_dir / "output_exacts"

        self.runtime_exact_file = self.base_dir / "runtime_exacte.txt"
        
        # TKUSP Versions V1 to V6
        self.version_dirs = {
            f"TKUSP_V{i}": self.base_dir / f"filesJSON{i}"
            for i in range(1, 9)
        }
        
        self.available_versions = [v for v, p in self.version_dirs.items() if p.exists()]
        
        # Colors and markers
        self.colors = {
            'TKUSP_V1': '#E63946', # Red
            'TKUSP_V2': '#2E86AB', # Blue
            'TKUSP_V3': '#06A77D', # Green
            'TKUSP_V4': '#F77F00', # Orange
            'TKUSP_V5': '#6D597A', # Purple
            'TKUSP_V6': '#D62828', # Dark Red
            'TKUSP_V7': '#0000FF', # Dark Blue
            'TKUSP_V8': '#FF00FF', # Magenta
            'HUSP-SP': '#000000'   # Black
        }
        
        self.markers = {
            'TKUSP_V1': 'o',
            'TKUSP_V2': 's',
            'TKUSP_V3': '^',
            'TKUSP_V4': 'D',
            'TKUSP_V5': 'v',
            'TKUSP_V6': 'P',
            'TKUSP_V7': 'x',
            'TKUSP_V8': 'H',
            'HUSP-SP': '*'
        }
        
        self.linestyles = {
            'TKUSP_V1': '-',
            'TKUSP_V2': '-',
            'TKUSP_V3': '-',
            'TKUSP_V4': '-',
            'TKUSP_V5': '-',
            'TKUSP_V6': '-',
            'TKUSP_V7': '-',
            'TKUSP_V8': '-',
            'HUSP-SP': '--'
        }

    def parse_runtime_exact(self) -> Dict[str, Dict[int, float]]:
        """Parses runtime_exacte.txt to get HUSP-SP runtimes."""
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
        Parses output files to get Avg Utility and Pattern Count for HUSP-SP.
        Returns: {k: {'avgUtil': float, 'count': int}}
        """
        results = {}
        dataset_dir = self.output_exact_dir / dataset
        if not dataset_dir.exists():
            return results
            
        # Look for files like SIGN_10.txt
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

    def load_heuristic_data(self, version: str, dataset: str) -> Dict[int, Dict[str, float]]:
        """
        Loads acc.json, runtime.json, avgUtil.json for a version/dataset.
        Returns: {k: {'accuracy': val, 'runtime': val, 'avgUtil': val}}
        """
        data = {}
        version_dir = self.version_dirs.get(version)
        if not version_dir:
            return data
            
        dataset_dir = version_dir / dataset
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
                                # Map JSON key to our internal key
                                # acc.json -> "accuracy"
                                # runtime.json -> "time" or "runtime"
                                # avgUtil.json -> "avgUtility"
                                
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
    # Union of datasets in runtime_exacte and directories in filesJSON1
    datasets = set(exact_runtimes.keys())
    if parser.version_dirs['TKUSP_V1'].exists():
        for d in parser.version_dirs['TKUSP_V1'].iterdir():
            if d.is_dir():
                datasets.add(d.name)
    
    # Filter datasets to only those we care about (optional, but good practice)
    # The user listed: SIGN, Leviathan, Yoochoose, BIBLE, Kosarak10k
    # We will process all found datasets that have data.
    
    output_dir = Path("plots_svg_comparison")
    output_dir.mkdir(exist_ok=True)
    
    for dataset in sorted(datasets):
        print(f"Processing {dataset}...")
        
        # Load Exact Data
        exact_results = parser.get_exact_results(dataset) # {k: {avgUtil, count}}
        exact_runtime_map = exact_runtimes.get(dataset, {}) # {k: runtime}
        
        # Load Heuristic Data
        heuristics_data = {}
        for v in parser.available_versions:
            heuristics_data[v] = parser.load_heuristic_data(v, dataset)
            
        # Collect all K values
        all_ks = set(exact_results.keys())
        all_ks.update(exact_runtime_map.keys())
        for v_data in heuristics_data.values():
            all_ks.update(v_data.keys())
            
        if not all_ks:
            print(f"  No data for {dataset}, skipping.")
            continue
            
        sorted_ks = sorted(list(all_ks))
        
        # Setup Plot
        fig, (ax_acc, ax_rt, ax_avg) = plt.subplots(1, 3, figsize=(24, 6))
        
        # --- 1. Accuracy Plot ---
        # HUSP-SP is always 100% (1.0)
        # We only plot HUSP-SP line at 100%
        ax_acc.axhline(y=100, color=parser.colors['HUSP-SP'], linestyle=parser.linestyles['HUSP-SP'], label='HUSP-SP', linewidth=2)
        
        for v in parser.available_versions:
            v_data = heuristics_data[v]
            xs = []
            ys = []
            for k in sorted_ks:
                if k in v_data and 'accuracy' in v_data[k]:
                    # Calculate percentage
                    # Accuracy in JSON is count of patterns found
                    # We need exact count for this k
                    exact_count = exact_results.get(k, {}).get('count', 0)
                    if exact_count > 0:
                        acc_pct = (v_data[k]['accuracy'] / exact_count) * 100
                        xs.append(k)
                        ys.append(acc_pct)
            
            if xs:
                ax_acc.plot(xs, ys, marker=parser.markers[v], color=parser.colors[v], 
                            linestyle=parser.linestyles[v], label=v, linewidth=1.5)
        
        ax_acc.set_title(f"{dataset} - Accuracy")
        ax_acc.set_xlabel("k")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.set_ylim(0, 100)
        ax_acc.grid(True, linestyle='--', alpha=0.5)
        ax_acc.legend()
        
        # --- 2. Runtime Plot ---
        # HUSP-SP
        ex_xs = []
        ex_ys = []
        for k in sorted_ks:
            if k in exact_runtime_map:
                ex_xs.append(k)
                ex_ys.append(exact_runtime_map[k])
        if ex_xs:
            ax_rt.plot(ex_xs, ex_ys, marker=parser.markers['HUSP-SP'], color=parser.colors['HUSP-SP'], 
                       linestyle=parser.linestyles['HUSP-SP'], label='HUSP-SP', linewidth=2)
            
        for v in parser.available_versions:
            v_data = heuristics_data[v]
            xs = []
            ys = []
            for k in sorted_ks:
                if k in v_data and 'runtime' in v_data[k]:
                    xs.append(k)
                    ys.append(v_data[k]['runtime'])
            if xs:
                ax_rt.plot(xs, ys, marker=parser.markers[v], color=parser.colors[v], 
                            linestyle=parser.linestyles[v], label=v, linewidth=1.5)

        ax_rt.set_title(f"{dataset} - Runtime")
        ax_rt.set_xlabel("k")
        ax_rt.set_ylabel("Time (s)")
        ax_rt.grid(True, linestyle='--', alpha=0.5)
        # ax_rt.set_yscale('log') # Optional: Uncomment if log scale is needed
        ax_rt.legend()

        # --- 3. Average Utility Plot ---
        # HUSP-SP
        ex_xs = []
        ex_ys = []
        for k in sorted_ks:
            if k in exact_results:
                ex_xs.append(k)
                ex_ys.append(exact_results[k]['avgUtil'])
        if ex_xs:
            ax_avg.plot(ex_xs, ex_ys, marker=parser.markers['HUSP-SP'], color=parser.colors['HUSP-SP'], 
                       linestyle=parser.linestyles['HUSP-SP'], label='HUSP-SP', linewidth=2)

        for v in parser.available_versions:
            v_data = heuristics_data[v]
            xs = []
            ys = []
            for k in sorted_ks:
                if k in v_data and 'avgUtil' in v_data[k]:
                    xs.append(k)
                    ys.append(v_data[k]['avgUtil'])
            if xs:
                ax_avg.plot(xs, ys, marker=parser.markers[v], color=parser.colors[v], 
                            linestyle=parser.linestyles[v], label=v, linewidth=1.5)

        ax_avg.set_title(f"{dataset} - Average Utility")
        ax_avg.set_xlabel("k")
        ax_avg.set_ylabel("Utility")
        ax_avg.grid(True, linestyle='--', alpha=0.5)
        ax_avg.legend()
        
        # Save
        plt.tight_layout()
        outfile = output_dir / f"{dataset}_comparison.svg"
        plt.savefig(outfile, format='svg')
        plt.close(fig)
        print(f"  Saved {outfile}")

if __name__ == "__main__":
    generate_plots()
