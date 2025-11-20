# AnkiMedBench Setup Guide

This guide provides detailed instructions for setting up and running AnkiMedBench.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Dataset Preparation](#dataset-preparation)
3. [Running Your First Benchmark](#running-your-first-benchmark)
4. [Understanding Results](#understanding-results)

## Environment Setup

### 1. System Requirements

- **OS**: Linux, macOS, or Windows with WSL2
- **Python**: 3.8 or higher
- **GPU**: CUDA-compatible GPU with 8GB+ VRAM (recommended for larger models)
- **RAM**: 16GB+ recommended
- **Storage**: 50GB+ for models and datasets

### 2. Python Environment

We recommend using conda or venv:

```bash
# Using conda
conda create -n ankimedbench python=3.9
conda activate ankimedbench

# Using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate
pip install pandas numpy scipy scikit-learn
pip install matplotlib seaborn
pip install tqdm
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

### Step 1: Obtain Medical Flashcards

Since Anki flashcards are copyrighted, you must obtain your own:

**Option A: Create Your Own**
1. Install Anki desktop application
2. Create or import medical flashcards
3. Export deck: File → Export → Notes in Plain Text

**Option B: Use Licensed Decks**
- Ensure you have proper licensing
- Common sources: AnKing, Zanki (check their licenses)

### Step 2: Prepare Flashcard Data

Once you have exported flashcards:

```bash
# Create data directory
mkdir -p data/anki_cards

# Your exported file should contain tab-separated values
# Process it to create front/back text files
```

Example processing script:

```python
# process_anki_export.py
import pandas as pd

# Read exported file
df = pd.read_csv('exported_deck.txt', sep='\t', header=None)
df.columns = ['front', 'back', 'tags']

# Save front and back separately
df['front'].to_csv('data/anki_cards/front_text.txt', index=False, header=False)
df['back'].to_csv('data/anki_cards/back_text.txt', index=False, header=False)
```

### Step 3: Download Benchmark Datasets

**BIOSSES** (Biomedical Semantic Similarity):

```bash
cd data
wget https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.zip
unzip DataSet.zip -d biosses/
```

**SciTail** (Science Entailment):

```python
from datasets import load_dataset
dataset = load_dataset('scitail', 'snli_format')
dataset.save_to_disk('data/scitail/')
```

**PubMedQA**:

```bash
cd data
mkdir pubmedqa
cd pubmedqa
wget https://github.com/pubmedqa/pubmedqa/raw/master/data/ori_pqaa.json
wget https://github.com/pubmedqa/pubmedqa/raw/master/data/ori_pqal.json
```

### Step 4: Verify Data Structure

Your `data/` directory should look like:

```
data/
├── anki_cards/
│   ├── front_text.txt    # ~10,000+ cards recommended
│   └── back_text.txt
├── biosses/
│   └── BIOSSES.txt
├── scitail/
│   ├── train/
│   ├── validation/
│   └── test/
└── pubmedqa/
    ├── ori_pqaa.json
    └── ori_pqal.json
```

## Running Your First Benchmark

### Quick Start: BIOSSES with BERT

```bash
cd benchmarks/BIOSSES

# Run with default BERT model
python run_biosses_benchmark.py \
    --model-name bert-base-uncased \
    --output-dir ../../results/biosses/

# With a medical-domain model
python run_biosses_benchmark.py \
    --model-name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --output-dir ../../results/biosses/
```

### Advanced: Running Multiple Models

```bash
# Run all BERT variants
for model in bert-base-uncased \
             microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
             dmis-lab/biobert-v1.1
do
    python run_biosses_benchmark.py \
        --model-name $model \
        --batch-size 32 \
        --output-dir ../../results/biosses/
done
```

### Memory-Constrained Environments

For systems with limited GPU memory:

```bash
# Use smaller batch size
python run_biosses_benchmark.py \
    --model-name bert-base-uncased \
    --batch-size 8 \
    --fp16  # Enable mixed precision
```

### Running SciTail and PubMedQA

```bash
# SciTail
cd benchmarks/SciTail
python run_scitail_benchmark.py --model-name bert-base-uncased

# PubMedQA
cd benchmarks/PubMedQA
python run_pubmedqa_benchmark.py --model-name bert-base-uncased
```

## Understanding Results

### Output Format

Each benchmark produces CSV files with results:

```csv
model_name,task,metric,score,num_samples,timestamp
bert-base-uncased,BIOSSES,pearson,0.8521,100,2024-01-01T12:00:00
bert-base-uncased,BIOSSES,spearman,0.8432,100,2024-01-01T12:00:00
```

### Analyzing Results

```bash
# Compile all results
python scripts/analysis/extract_anki_results.py

# Generate statistical analysis
python scripts/analysis/analyze_anki_results.py \
    --results-dir results/ \
    --output-dir analysis/

# Create visualizations
python scripts/visualization/visualize_anki_results.py \
    --results-file results/compiled_results.csv \
    --output-dir figures/
```

### Interpreting Metrics

**BIOSSES**:
- Pearson correlation: Linear relationship between predicted and true scores
- Spearman correlation: Monotonic relationship (rank-based)
- Target: >0.80 is considered good for biomedical similarity

**SciTail**:
- Accuracy: Percentage of correct entailment predictions
- F1-score: Harmonic mean of precision and recall
- Target: >0.85 is competitive

**PubMedQA**:
- Accuracy: Correct answer selection
- Exact Match: Percentage of perfectly matched answers
- Target: >0.70 is strong performance

## Troubleshooting

### Common Issues

**Out of Memory (OOM) Errors**:
```bash
# Reduce batch size
--batch-size 4

# Use CPU instead of GPU
--device cpu

# Enable gradient checkpointing (if supported)
--gradient-checkpointing
```

**Model Download Fails**:
```bash
# Pre-download models
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

**Missing Dependencies**:
```bash
pip install --upgrade transformers datasets
```

## Next Steps

1. Run benchmarks on multiple models
2. Compare results across tasks
3. Fine-tune models on your specific data
4. Contribute improvements back to the project

For more information, see the main [README](../README.md) or open an issue on GitHub.
