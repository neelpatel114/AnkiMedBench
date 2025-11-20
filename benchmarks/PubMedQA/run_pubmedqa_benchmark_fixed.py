#!/usr/bin/env python3
"""
Proper PubMedQA Benchmarking for Gemma Models
Following best practices for embedding model evaluation
"""

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
import json
from pathlib import Path
from datetime import datetime
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import os
warnings.filterwarnings('ignore')

class ProperPubMedQABenchmark:
    """Comprehensive PubMedQA benchmark following best practices"""
    
    def __init__(self, gpu_id=1):
        # Use GPU 1 to avoid conflict with other session
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU {gpu_id}: {torch.cuda.get_device_name()}")
    
    def load_pubmedqa_data(self, data_path):
        """Load PubMedQA dataset"""
        print("\n" + "="*70)
        print("Loading PubMedQA Dataset")
        print("="*70)
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Process questions and contexts
        questions = [item['question'] for item in data]
        contexts = [item['context'] for item in data]
        answers = [item['answer'] for item in data]
        
        print(f"✓ Loaded {len(data)} QA pairs")
        print(f"✓ Answer distribution: yes={sum(1 for a in answers if a=='yes')}, no={sum(1 for a in answers if a=='no')}, maybe={sum(1 for a in answers if a=='maybe')}")
        print(f"✓ Avg context length: {np.mean([len(c.split()) for c in contexts]):.0f} words")
        
        return questions, contexts, answers
    
    def mean_pooling(self, model_output, attention_mask):
        """Standard mean pooling"""
        token_embeddings = model_output.hidden_states[-1]  # Use last hidden state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def generate_embeddings(self, model, tokenizer, texts, batch_size=4):
        """Generate normalized embeddings for Gemma using mean pooling"""
        model.eval()
        all_embeddings = []
        
        # Ensure tokenizer has padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                outputs = model(**inputs, output_hidden_states=True)
                # Use mean pooling
                embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def calculate_retrieval_metrics(self, query_embeddings, doc_embeddings, k_values=[1, 3, 5, 10]):
        """Calculate retrieval metrics"""
        # Compute similarity matrix
        similarities = cosine_similarity(query_embeddings, doc_embeddings)
        
        metrics = {}
        
        # For each query, the relevant document is at the same index
        n_queries = len(query_embeddings)
        
        # Precision@K and Recall@K
        for k in k_values:
            precision_scores = []
            recall_scores = []
            
            for i in range(n_queries):
                # Get top-k indices
                top_k_indices = np.argsort(similarities[i])[-k:][::-1]
                
                # Check if correct document is in top-k
                if i in top_k_indices:
                    precision_scores.append(1.0 / k)
                    recall_scores.append(1.0)  # Since there's only 1 relevant doc
                else:
                    precision_scores.append(0.0)
                    recall_scores.append(0.0)
            
            metrics[f'precision@{k}'] = float(np.mean(precision_scores))
            metrics[f'recall@{k}'] = float(np.mean(recall_scores))
        
        # Mean Reciprocal Rank (MRR)
        mrr_scores = []
        for i in range(n_queries):
            ranked_indices = np.argsort(similarities[i])[::-1]
            rank = np.where(ranked_indices == i)[0][0] + 1
            mrr_scores.append(1.0 / rank)
        metrics['mrr'] = float(np.mean(mrr_scores))
        
        # Mean Average Precision (MAP)
        map_scores = []
        for i in range(n_queries):
            ranked_indices = np.argsort(similarities[i])[::-1]
            rank = np.where(ranked_indices == i)[0][0] + 1
            if rank <= 10:  # Consider only top-10
                map_scores.append(1.0 / rank)
            else:
                map_scores.append(0.0)
        metrics['map@10'] = float(np.mean(map_scores))
        
        # NDCG@10
        ndcg_scores = []
        for i in range(n_queries):
            ranked_indices = np.argsort(similarities[i])[-10:][::-1]
            relevance = [1 if idx == i else 0 for idx in ranked_indices]
            
            # Calculate DCG
            dcg = relevance[0]
            for j in range(1, len(relevance)):
                dcg += relevance[j] / np.log2(j + 2)
            
            # Calculate IDCG (ideal DCG)
            ideal_relevance = sorted(relevance, reverse=True)
            idcg = ideal_relevance[0]
            for j in range(1, len(ideal_relevance)):
                idcg += ideal_relevance[j] / np.log2(j + 2)
            
            # Calculate NDCG
            if idcg > 0:
                ndcg_scores.append(dcg / idcg)
            else:
                ndcg_scores.append(0.0)
        
        metrics['ndcg@10'] = float(np.mean(ndcg_scores))
        
        return metrics
    
    def benchmark_model(self, model_path, model_name, data_path):
        """Run PubMedQA benchmark for a single model"""
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
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # Load data
        questions, contexts, answers = self.load_pubmedqa_data(data_path)
        
        # Generate embeddings
        print("Generating embeddings (using mean pooling)...")
        query_embeddings = self.generate_embeddings(model, tokenizer, questions)
        context_embeddings = self.generate_embeddings(model, tokenizer, contexts)
        
        # Calculate retrieval metrics
        print("Calculating retrieval metrics...")
        retrieval_metrics = self.calculate_retrieval_metrics(query_embeddings, context_embeddings)
        
        print(f"\nResults:")
        print(f"  MRR:     {retrieval_metrics['mrr']:.4f}")
        print(f"  MAP@10:  {retrieval_metrics['map@10']:.4f}")
        print(f"  NDCG@10: {retrieval_metrics['ndcg@10']:.4f}")
        print(f"  Recall@10: {retrieval_metrics['recall@10']:.4f}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
        return retrieval_metrics

def main():
    """Run PubMedQA benchmark for Gemma models"""
    
    # Use absolute paths on server
    base_path = Path("/home/AD.UNLV.EDU/pateln3/thesis_models/thesis_final/model_complete_structure")
    
    GEMMA_MODELS = {
        "Gemma-base": str(base_path / "models" / "Gemma2-9B" / "models" / "base"),
        "Gemma-raw": str(base_path / "models" / "Gemma2-9B" / "models" / "raw"),
        "Gemma-enhanced": str(base_path / "models" / "Gemma2-9B" / "models" / "enhanced")
    }
    
    DATA_PATH = str(base_path / "benchmark_datasets" / "pubmedqa_data" / "pubmedqa.json")
    
    # Initialize benchmark
    benchmark = ProperPubMedQABenchmark(gpu_id=1)
    
    # Run for each Gemma model
    all_results = {}
    
    for model_name, model_path in GEMMA_MODELS.items():
        results = benchmark.benchmark_model(model_path, model_name, DATA_PATH)
        all_results[model_name] = results
    
    # Comparative analysis
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS - Gemma Models on PubMedQA")
    print("="*70)
    
    print("\nRetrieval Performance:")
    print("-" * 40)
    print(f"{'Model':15} {'MRR':8} {'MAP@10':8} {'NDCG@10':8} {'Recall@10':10}")
    print("-" * 40)
    for model_name in GEMMA_MODELS.keys():
        mrr = all_results[model_name]['mrr']
        map_score = all_results[model_name]['map@10']
        ndcg = all_results[model_name]['ndcg@10']
        recall = all_results[model_name]['recall@10']
        print(f"{model_name:15} {mrr:.4f}   {map_score:.4f}   {ndcg:.4f}   {recall:.4f}")
    
    # Calculate improvements
    base_mrr = all_results['Gemma-base']['mrr']
    raw_mrr = all_results['Gemma-raw']['mrr']
    enhanced_mrr = all_results['Gemma-enhanced']['mrr']
    
    print("\nImprovements (MRR):")
    print("-" * 40)
    print(f"Finetuning effect (base → raw):      {(raw_mrr - base_mrr):.4f} "
          f"({((raw_mrr - base_mrr) / base_mrr * 100):+.1f}%)")
    print(f"Enhancement effect (raw → enhanced):  {(enhanced_mrr - raw_mrr):.4f} "
          f"({((enhanced_mrr - raw_mrr) / raw_mrr * 100):+.1f}%)")
    print(f"Total improvement (base → enhanced): {(enhanced_mrr - base_mrr):.4f} "
          f"({((enhanced_mrr - base_mrr) / base_mrr * 100):+.1f}%)")
    
    # Save results
    output_path = Path("/home/AD.UNLV.EDU/pateln3/thesis_models/thesis_final/model_complete_structure/models/Gemma2-9B/benchmarks/PubMedQA")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary
    summary = {
        "benchmark": "PubMedQA",
        "timestamp": timestamp,
        "model_type": "Gemma",
        "models": all_results,
        "improvements": {
            "finetuning_effect_mrr": float(raw_mrr - base_mrr),
            "enhancement_effect_mrr": float(enhanced_mrr - raw_mrr),
            "total_improvement_mrr": float(enhanced_mrr - base_mrr)
        }
    }
    
    with open(output_path / f"pubmedqa_results_{timestamp}.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Results saved to Gemma/benchmarks/PubMedQA/pubmedqa_results_{timestamp}.json")
    print("✓ PubMedQA Benchmarking Complete")

if __name__ == "__main__":
    main()
