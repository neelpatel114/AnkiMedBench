# AnkiMedBench

A comprehensive benchmark suite for evaluating medical embedding models using Anki flashcard-based datasets.

## Overview

AnkiMedBench is a benchmarking framework designed to evaluate the performance of various embedding models on medical domain tasks. The benchmark uses medically curated Anki flashcards as the foundation for creating challenging evaluation datasets across multiple biomedical NLP tasks.

### Key Features

- **Multiple Benchmark Tasks**: BIOSSES, SciTail, and PubMedQA
- **Model Agnostic**: Supports BERT, ModernBERT, Gemma, Llama, and Qwen families
- **Comprehensive Analysis**: Statistical analysis and visualization tools included
- **Reproducible**: Detailed instructions for dataset preparation and evaluation

## Supported Models

AnkiMedBench has been tested with the following model families:

- **BERT**: Base and medical domain-adapted variants
- **ModernBERT**: Latest BERT architecture improvements
- **Gemma**: Google's lightweight models (2B, 7B, 9B)
- **Llama**: Meta's LLaMA models
- **Qwen**: Alibaba's Qwen models

## Project Structure

```
AnkiMedBench/
├── benchmarks/           # Benchmark task implementations
│   ├── BIOSSES/         # Biomedical semantic similarity
│   ├── SciTail/         # Scientific entailment
│   └── PubMedQA/        # PubMed question answering
├── scripts/
│   ├── analysis/        # Result analysis scripts
│   └── visualization/   # Plotting and visualization
├── data/
│   └── sample/          # Sample data structure (see below)
├── results/             # Benchmark results (generated)
└── docs/                # Additional documentation

```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- pandas, numpy, matplotlib, seaborn
- scikit-learn

### Setup

```bash
# Clone the repository
git clone git@github.com:neelpatel114/AnkiMedBench.git
cd AnkiMedBench

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

**Important**: Due to copyright restrictions, the Anki flashcard datasets cannot be distributed directly. You must prepare your own medical Anki decks.

### Data Structure

Your data should follow this structure:

```
data/
├── anki_cards/              # Your Anki flashcards (not included)
│   ├── front_text.txt      # Questions/prompts
│   └── back_text.txt       # Answers/explanations
├── biosses/                 # BIOSSES dataset
├── scitail/                 # SciTail dataset
└── pubmedqa/                # PubMedQA dataset
```

### Obtaining Medical Anki Decks

1. **Create or obtain** medical Anki flashcards (ensure proper licensing)
2. **Export** your Anki deck to text format
3. **Place** the exported data in `data/anki_cards/`

Popular medical Anki resources (check licensing):
- AnKing Medical Deck
- Zanki Step Decks
- Custom medical school decks

### Preparing Benchmark Datasets

The benchmark uses standard biomedical NLP datasets:

```bash
# Download BIOSSES
wget https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.zip -P data/

# SciTail is available through HuggingFace datasets
python -c "from datasets import load_dataset; load_dataset('scitail', 'snli_format')"

# PubMedQA
wget https://github.com/pubmedqa/pubmedqa/raw/master/data/ori_pqaa.json -P data/
```

## Usage

### Running Benchmarks

```bash
# Run BIOSSES benchmark with a specific model
cd benchmarks/BIOSSES
python run_biosses_benchmark.py --model-name bert-base-uncased

# Run SciTail benchmark
cd benchmarks/SciTail
python run_scitail_benchmark.py --model-name google/gemma-2b

# Run PubMedQA benchmark
cd benchmarks/PubMedQA
python run_pubmedqa_benchmark.py --model-name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract
```

### Analyzing Results

```bash
# Extract and compile results
python scripts/analysis/extract_anki_results.py

# Generate analysis
python scripts/analysis/analyze_anki_results.py

# Create visualizations
python scripts/visualization/visualize_anki_results.py
```

## Results Format

Results are saved in CSV format with the following structure:

```csv
model_name,task,metric,score,timestamp
bert-base-uncased,BIOSSES,pearson_correlation,0.85,2024-01-01
```

## Citation

If you use AnkiMedBench in your research, please cite:

```bibtex
@misc{ankimedbench2024,
  title={AnkiMedBench: A Benchmark for Medical Embedding Models},
  author={Your Name},
  year={2024},
  url={https://github.com/neelpatel114/AnkiMedBench}
}
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - See LICENSE file for details

## Acknowledgments

- BIOSSES dataset creators
- SciTail dataset from Allen AI
- PubMedQA dataset contributors
- HuggingFace for model hosting

## Contact

For questions or issues, please open a GitHub issue or contact [your email].

---

**Note**: This benchmark framework does not include copyrighted Anki flashcard content. Users must obtain and prepare their own medical flashcard datasets according to applicable licenses and terms of use.
