#!/usr/bin/env python3
"""
SciTail Benchmarking for ModernBERT Models
Test-only evaluation on scientific textual entailment using local data
"""

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

class ModernBERTSciTailBenchmark:
    """SciTail benchmark for ModernBERT models - test set evaluation only"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
    
    def load_scitail_data(self, data_path):
        """Load SciTail data from local JSON file"""
        print("\n" + "="*70)
        print("Loading SciTail Dataset")
        print("="*70)
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        print(f"✓ Loaded {len(data)} samples")
        
        # Check label distribution
        label_counts = {}
        for item in data:
            label = item['label']
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
        
        print("\nLabel distribution:")
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count} ({count/len(data)*100:.1f}%)")
        
        return data
    
    def mean_pooling(self, model_output, attention_mask):
        """Standard mean pooling for embeddings"""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def encode_batch(self, model, tokenizer, texts, batch_size=32, max_length=256):
        """Encode multiple texts in batches"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().numpy())
            
            # Clear cache periodically
            if (i // batch_size) % 10 == 0:
                torch.cuda.empty_cache()
        
        return np.vstack(all_embeddings)
    
    def evaluate_similarity_based(self, premise_embs, hypothesis_embs, labels):
        """Evaluate using cosine similarity with multiple thresholds"""
        # Compute cosine similarities
        similarities = np.sum(premise_embs * hypothesis_embs, axis=1)
        
        # Convert labels to binary (entailment=1, neutral=0)
        binary_labels = np.array([1 if l == 'entailment' else 0 for l in labels])
        
        # Find optimal threshold by trying multiple values
        thresholds = np.percentile(similarities, [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75])
        best_accuracy = 0
        best_threshold = 0
        best_predictions = None
        
        for threshold in thresholds:
            predictions = (similarities > threshold).astype(int)
            accuracy = accuracy_score(binary_labels, predictions)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
                best_predictions = predictions
        
        # Calculate metrics with best threshold
        accuracy = accuracy_score(binary_labels, best_predictions)
        f1 = f1_score(binary_labels, best_predictions)
        
        # Calculate AUC using raw similarities
        auc = None
        if len(np.unique(binary_labels)) > 1:
            try:
                auc = roc_auc_score(binary_labels, similarities)
            except:
                auc = None
        
        # Get classification report
        report = classification_report(binary_labels, best_predictions, 
                                      target_names=['neutral', 'entailment'],
                                      output_dict=True)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'auc_roc': auc,
            'optimal_threshold': best_threshold,
            'classification_report': report,
            'similarities_mean': float(np.mean(similarities)),
            'similarities_std': float(np.std(similarities))
        }
    
    def benchmark_model(self, model_path, model_name, data_path):
        """Run SciTail benchmark for a single model"""
        print(f"\n{'='*70}")
        print(f"Benchmarking: {model_name}")
        print('='*70)
        
        try:
            # Load model and tokenizer
            print("Loading model...")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token
            
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            ).to(self.device)
            
            model.eval()
            
            # Load test data
            test_data = self.load_scitail_data(data_path)
            
            print(f"\nEvaluating on {len(test_data)} samples...")
            
            # Extract premises, hypotheses, and labels
            premises = [item['premise'] for item in test_data]
            hypotheses = [item['hypothesis'] for item in test_data]
            labels = [item['label'] for item in test_data]
            
            # Encode premises and hypotheses
            print("Encoding premises...")
            premise_embs = self.encode_batch(model, tokenizer, premises)
            
            print("Encoding hypotheses...")
            hypothesis_embs = self.encode_batch(model, tokenizer, hypotheses)
            
            # Evaluate
            print("\nEvaluating performance...")
            results = self.evaluate_similarity_based(
                premise_embs, hypothesis_embs, labels
            )
            
            # Display results
            print("\n" + "="*70)
            print("RESULTS")
            print("="*70)
            print(f"Accuracy: {results['accuracy']:.4f}")
            print(f"F1 Score: {results['f1_score']:.4f}")
            if results['auc_roc']:
                print(f"AUC-ROC: {results['auc_roc']:.4f}")
            print(f"Optimal Threshold: {results['optimal_threshold']:.4f}")
            print(f"Mean Similarity: {results['similarities_mean']:.4f} (±{results['similarities_std']:.4f})")
            
            print("\nPer-class Performance:")
            for class_name in ['neutral', 'entailment']:
                if class_name in results['classification_report']:
                    class_results = results['classification_report'][class_name]
                    print(f"  {class_name:12} - Precision: {class_results['precision']:.3f}, "
                          f"Recall: {class_results['recall']:.3f}, "
                          f"F1: {class_results['f1-score']:.3f}")
            
            # Clean up
            del model
            torch.cuda.empty_cache()
            
            return results
            
        except Exception as e:
            print(f"Error benchmarking {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Run SciTail benchmark for ModernBERT models"""
    
    # Use absolute paths on server
    base_path = Path("/home/AD.UNLV.EDU/pateln3/thesis_models/thesis_final/model_complete_structure")
    
    MODERNBERT_MODELS = {
        "ModernBERT-base": str(base_path / "models" / "ModernBERT" / "models" / "base"),
        "ModernBERT-raw": str(base_path / "models" / "ModernBERT" / "models" / "simcse/raw"),
        "ModernBERT-enhanced": str(base_path / "models" / "ModernBERT" / "models" / "simcse/enhanced")
    }
    
    # Use local data file - use sample for faster testing
    DATA_PATH = str(base_path / "benchmark_datasets" / "scitail_data" / "scitail_sample.json")
    
    # Initialize benchmark
    benchmark = ModernBERTSciTailBenchmark()
    
    # Run for each ModernBERT model
    all_results = {}
    
    for model_name, model_path in MODERNBERT_MODELS.items():
        results = benchmark.benchmark_model(model_path, model_name, DATA_PATH)
        if results:
            all_results[model_name] = results
    
    if not all_results:
        print("\nNo successful results to save")
        return
    
    # Comparative analysis
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS - ModernBERT Models on SciTail")
    print("="*70)
    
    print(f"\n{'Model':<20} {'Accuracy':<12} {'F1 Score':<12} {'AUC-ROC':<12}")
    print("-" * 56)
    
    for model_name in MODERNBERT_MODELS.keys():
        if model_name in all_results:
            res = all_results[model_name]
            auc_str = f"{res['auc_roc']:.4f}" if res['auc_roc'] else "N/A"
            print(f"{model_name:<20} {res['accuracy']:.4f}      {res['f1_score']:.4f}      {auc_str}")
    
    # Calculate improvements if we have all models
    if len(all_results) >= 2:
        print("\nModel Improvements:")
        print("-" * 40)
        
        if 'ModernBERT-base' in all_results and 'ModernBERT-raw' in all_results:
            base_acc = all_results['ModernBERT-base']['accuracy']
            raw_acc = all_results['ModernBERT-raw']['accuracy']
            print(f"Finetuning effect (base→raw): {(raw_acc - base_acc):+.4f}")
            
            if 'ModernBERT-enhanced' in all_results:
                enhanced_acc = all_results['ModernBERT-enhanced']['accuracy']
                print(f"Enhancement effect (raw→enhanced): {(enhanced_acc - raw_acc):+.4f}")
                print(f"Total improvement (base→enhanced): {(enhanced_acc - base_acc):+.4f}")
    
    # Save results
    output_path = Path(__file__).parent
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare JSON-serializable summary
    summary = {
        "benchmark": "SciTail",
        "timestamp": timestamp,
        "dataset_info": {
            "task": "Textual Entailment",
            "data_file": "scitail_sample.json",
            "labels": ["entailment", "neutral"],
            "domain": "Scientific Text",
            "evaluation": "Similarity-based classification"
        },
        "models": {}
    }
    
    for model_name, results in all_results.items():
        # Remove non-serializable parts
        results_copy = results.copy()
        results_copy.pop('classification_report', None)
        summary["models"][model_name] = results_copy
    
    # Calculate improvements for summary
    if 'ModernBERT-base' in all_results and 'ModernBERT-raw' in all_results:
        base_acc = all_results['ModernBERT-base']['accuracy']
        raw_acc = all_results['ModernBERT-raw']['accuracy']
        summary["improvements"] = {
            "finetuning_effect": float(raw_acc - base_acc)
        }
        if 'ModernBERT-enhanced' in all_results:
            enhanced_acc = all_results['ModernBERT-enhanced']['accuracy']
            summary["improvements"]["enhancement_effect"] = float(enhanced_acc - raw_acc)
            summary["improvements"]["total_improvement"] = float(enhanced_acc - base_acc)
    
    with open(output_path / "results" / "simcse" / f"scitail_results_{timestamp}.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Results saved to scitail_results_{timestamp}.json")
    print("✓ SciTail Benchmarking Complete for ModernBERT")

if __name__ == "__main__":
    main()
