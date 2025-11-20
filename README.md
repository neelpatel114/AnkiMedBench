# AnkiMedBench: Evaluating Hierarchical Medical Knowledge in Language Model Embeddings

A comprehensive benchmark for evaluating how well language models preserve hierarchical medical knowledge structures.

## Overview

AnkiMedBench evaluates whether embedding models preserve the taxonomic structure essential for clinical reasoning. Built from 16,512 medical flashcards spanning 6 hierarchy levels—from 16 broad specialties to 672 specific diseases—it measures both exact classification (Flat F1) and hierarchical coherence (Hierarchical F1).

Despite 90%+ accuracy on medical benchmarks, models often fail at hierarchical navigation critical for clinical practice. AnkiMedBench reveals these invisible limitations by testing whether models can traverse diagnostic hierarchies: from chest pain → cardiovascular pathology → myocardial infarction → specific STEMI types.

### Key Features

- **Hierarchical Evaluation**: Dual-metric approach (Flat F1 + Hierarchical F1) reveals model quality
- **16,512 Medical Flashcards**: Organized across 6 hierarchy levels from specialties to specific conditions
- **30 Models Tested**: BERT, ModernBERT, Gemma, Llama, Qwen families evaluated
- **Clinical Relevance**: Tests taxonomic navigation essential for diagnostic reasoning
- **Reproducible Framework**: Detailed instructions for dataset preparation and evaluation

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
@mastersthesis{patel2025ankimedbench,
  title={AnkiMedBench: Evaluating Hierarchical Medical Knowledge in Language Model Embeddings},
  author={Patel, Neel},
  year={2025},
  school={University of Nevada, Las Vegas},
  type={Master's Thesis},
  url={https://github.com/neelpatel114/AnkiMedBench}
}
```

## Research Context

Current medical benchmarks test isolated factual recall but miss hierarchical reasoning. AnkiMedBench addresses this gap by:

- **Measuring Taxonomic Structure**: Tests whether models preserve diagnostic category hierarchies
- **Dual-Metric Approach**: Flat F1 for precision, Hierarchical F1 for structural coherence
- **Clinical Validity**: Based on flashcards from medical licensing exam preparation
- **Quality Detection**: High-quality models show minimal metric divergence; low-quality show large gaps

**Key Finding**: Our best model outperformed the worst by 2.6× at fine-grained classification. Fine-tuning on 26 medical textbooks produced minimal hierarchical improvement, demonstrating that standard benchmark performance does not predict hierarchical capability.

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
