import re
import json
import os
from pathlib import Path
from typing import Dict, Any, List

class MetricsGenerator:
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.output_exact_dir = self.base_dir / "output_exacts"
        self.runtime_exact_file = self.base_dir / "runtime_exacte.txt"
        self.output_json_dir = self.base_dir / "filesJSON_TKHUSP"

    def parse_runtime_exact(self) -> Dict[str, Dict[int, Dict[str, float]]]:
        """Parses runtime_exacte.txt to get runtime and memory."""
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

    def get_exact_accuracy_utility(self, dataset: str) -> Dict[int, Dict[str, float]]:
        """Parses output files to get Avg Utility and Pattern Count."""
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
                    results[k] = {'avgUtil': avg_util, 'accuracy': len(utilities)}
                else:
                    results[k] = {'avgUtil': 0.0, 'accuracy': 0}
                    
            except Exception as e:
                print(f"Error parsing {file}: {e}")
                continue
                
        return results

    def save_json_files(self, dataset: str, k_values: List[int], 
                       runtime_data: Dict[int, Dict[str, float]], 
                       acc_util_data: Dict[int, Dict[str, float]]):
        """Saves metrics to separate JSON files."""
        dataset_out_dir = self.output_json_dir / dataset
        dataset_out_dir.mkdir(parents=True, exist_ok=True)
        
        acc_list = []
        avg_list = []
        rt_list = []
        mem_list = []
        
        for k in sorted(k_values):
            # Accuracy and Avg Util
            if k in acc_util_data:
                acc_list.append({"k": k, "accuracy": acc_util_data[k]['accuracy']})
                avg_list.append({"k": k, "avgUtility": acc_util_data[k]['avgUtil']})
            
            # Runtime and Memory
            if k in runtime_data:
                if 'runtime' in runtime_data[k]:
                    rt_list.append({"k": k, "runtime": runtime_data[k]['runtime']})
                if 'memory' in runtime_data[k]:
                    mem_list.append({"k": k, "memory": runtime_data[k]['memory']})
                    
        # Write files
        with open(dataset_out_dir / "acc.json", 'w') as f:
            json.dump(acc_list, f, indent=2)
        with open(dataset_out_dir / "avgUtil.json", 'w') as f:
            json.dump(avg_list, f, indent=2)
        with open(dataset_out_dir / "runtime.json", 'w') as f:
            json.dump(rt_list, f, indent=2)
        with open(dataset_out_dir / "memory.json", 'w') as f:
            json.dump(mem_list, f, indent=2)
            
        print(f"  ✅ Saved JSONs for {dataset}")

    def run(self):
        print("Starting generation of TKHUSP-Miner JSON metrics...")
        
        # 1. Parse Runtimes and Memory for ALL datasets
        all_runtimes = self.parse_runtime_exact()
        
        # 2. Identify all datasets from output_exacts directory
        # (This is more reliable than just relying on runtime file)
        datasets = set()
        if self.output_exact_dir.exists():
            for d in self.output_exact_dir.iterdir():
                if d.is_dir():
                    datasets.add(d.name)
        
        # Also include datasets found in runtime file
        datasets.update(all_runtimes.keys())
        
        for dataset in sorted(datasets):
            print(f"Processing {dataset}...")
            
            # Get Accuracy and Utility
            acc_util_data = self.get_exact_accuracy_utility(dataset)
            
            # Get Runtime and Memory for this dataset
            runtime_data = all_runtimes.get(dataset, {})
            
            # Collect all unique K values
            all_ks = set(acc_util_data.keys())
            all_ks.update(runtime_data.keys())
            
            if not all_ks:
                print(f"  ⚠️ No data found for {dataset}, skipping.")
                continue
                
            self.save_json_files(dataset, list(all_ks), runtime_data, acc_util_data)
            
        print("\nGeneration complete! Files saved in 'filesJSON_TKHUSP/'.")

if __name__ == "__main__":
    generator = MetricsGenerator()
    generator.run()
