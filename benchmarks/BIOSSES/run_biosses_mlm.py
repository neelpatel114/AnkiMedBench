#!/usr/bin/env python3
"""
BIOSSES Benchmarking for ModernBERT Models
Using proper mean pooling and evaluation methodology
"""

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
import json
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ModernBERTBIOSSESBenchmark:
    """BIOSSES benchmark for ModernBERT models"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
    
    def load_biosses_data(self, data_path):
        """Load BIOSSES dataset"""
        print("\n" + "="*70)
        print("Loading BIOSSES Dataset")
        print("="*70)
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        sentences1 = [item['sentence1'] for item in data]
        sentences2 = [item['sentence2'] for item in data]
        ground_truth_scores = [item['score'] for item in data]
        
        print(f"✓ Loaded {len(data)} sentence pairs")
        print(f"✓ Score range: {min(ground_truth_scores):.2f} to {max(ground_truth_scores):.2f}")
        print(f"✓ Mean score: {np.mean(ground_truth_scores):.2f} ± {np.std(ground_truth_scores):.2f}")
        
        return sentences1, sentences2, ground_truth_scores
    
    def mean_pooling(self, model_output, attention_mask):
        """Standard mean pooling"""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def generate_embeddings(self, model, tokenizer, sentences, batch_size=8):
        """Generate normalized embeddings"""
        model.eval()
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i+batch_size]
                
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                outputs = model(**inputs, output_hidden_states=True)
                embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def compute_similarity_scores(self, embeddings1, embeddings2):
        """Compute cosine similarity scores"""
        similarities = []
        for emb1, emb2 in zip(embeddings1, embeddings2):
            sim = np.dot(emb1, emb2)  # For normalized vectors, dot = cosine
            similarities.append(sim)
        return np.array(similarities)
    
    def calculate_correlations(self, predicted_scores, ground_truth_scores):
        """Calculate correlation metrics"""
        pearson_corr, pearson_p = pearsonr(predicted_scores, ground_truth_scores)
        spearman_corr, spearman_p = spearmanr(predicted_scores, ground_truth_scores)
        
        # Calculate confidence intervals
        n = len(predicted_scores)
        fisher_z = np.arctanh(pearson_corr)
        se = 1 / np.sqrt(n - 3)
        ci_lower = np.tanh(fisher_z - 1.96 * se)
        ci_upper = np.tanh(fisher_z + 1.96 * se)
        
        return {
            'pearson': {
                'correlation': float(pearson_corr),
                'p_value': float(pearson_p),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper)
            },
            'spearman': {
                'correlation': float(spearman_corr),
                'p_value': float(spearman_p)
            }
        }
    
    def benchmark_model(self, model_path, model_name, data_path):
        """Run BIOSSES benchmark for a single model"""
        print(f"\n{'='*70}")
        print(f"Benchmarking: {model_name}")
        print('='*70)
        
        # Load model and tokenizer
        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token
        
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(self.device)
        
        # Load data
        sentences1, sentences2, ground_truth_scores = self.load_biosses_data(data_path)
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings1 = self.generate_embeddings(model, tokenizer, sentences1)
        embeddings2 = self.generate_embeddings(model, tokenizer, sentences2)
        
        # Compute similarities
        predicted_scores = self.compute_similarity_scores(embeddings1, embeddings2)
        
        # Calculate correlations
        results = self.calculate_correlations(predicted_scores, ground_truth_scores)
        
        print(f"\nResults:")
        print(f"  Pearson:  {results['pearson']['correlation']:.4f} "
              f"(p={results['pearson']['p_value']:.2e}, "
              f"CI: [{results['pearson']['ci_lower']:.3f}, {results['pearson']['ci_upper']:.3f}])")
        print(f"  Spearman: {results['spearman']['correlation']:.4f} "
              f"(p={results['spearman']['p_value']:.2e})")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
        return results

def main():
    """Run BIOSSES benchmark for ModernBERT models"""
    
    # Use absolute paths on server
    base_path = Path("/home/AD.UNLV.EDU/pateln3/thesis_models/thesis_final/model_complete_structure")
    
    MODERNBERT_MODELS = {
        "ModernBERT-base": str(base_path / "models" / "ModernBERT" / "models" / "base"),
        "ModernBERT-MLM-raw": str(base_path / "models" / "ModernBERT" / "models" / "mlm/raw"),
        "ModernBERT-MLM-enhanced": str(base_path / "models" / "ModernBERT" / "models" / "mlm/enhanced")
    }
    
    DATA_PATH = str(base_path / "benchmark_datasets" / "biosses_data" / "biosses.json")
    
    # Initialize benchmark
    benchmark = ModernBERTBIOSSESBenchmark()
    
    # Run for each ModernBERT model
    all_results = {}
    
    for model_name, model_path in MODERNBERT_MODELS.items():
        results = benchmark.benchmark_model(model_path, model_name, DATA_PATH)
        all_results[model_name] = results
    
    # Comparative analysis
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS - ModernBERT Models on BIOSSES")
    print("="*70)
    
    print("\nPearson Correlations:")
    print("-" * 40)
    for model_name in MODERNBERT_MODELS.keys():
        corr = all_results[model_name]['pearson']['correlation']
        ci_lower = all_results[model_name]['pearson']['ci_lower']
        ci_upper = all_results[model_name]['pearson']['ci_upper']
        print(f"{model_name:20} {corr:.4f} (CI: [{ci_lower:.3f}, {ci_upper:.3f}])")
    
    # Calculate improvements
    base_pearson = all_results['ModernBERT-base']['pearson']['correlation']
    raw_pearson = all_results['ModernBERT-MLM-raw']['pearson']['correlation']
    enhanced_pearson = all_results['ModernBERT-MLM-enhanced']['pearson']['correlation']
    
    print("\nImprovements:")
    print("-" * 40)
    print(f"Finetuning effect (base → raw):      {(raw_pearson - base_pearson):.4f} "
          f"({((raw_pearson - base_pearson) / base_pearson * 100):+.1f}%)")
    print(f"Enhancement effect (raw → enhanced):  {(enhanced_pearson - raw_pearson):.4f} "
          f"({((enhanced_pearson - raw_pearson) / raw_pearson * 100):+.1f}%)")
    
    # Save results
    output_path = Path(__file__).parent
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary
    summary = {
        "benchmark": "BIOSSES",
        "timestamp": timestamp,
        "models": all_results,
        "improvements": {
            "finetuning_effect": float(raw_pearson - base_pearson),
            "enhancement_effect": float(enhanced_pearson - raw_pearson)
        }
    }
    
    with open(output_path / "results" / "mlm" / f"biosses_results_{timestamp}.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Results saved to biosses_results_{timestamp}.json")
    print("✓ BIOSSES Benchmarking Complete for ModernBERT")

if __name__ == "__main__":
    main()