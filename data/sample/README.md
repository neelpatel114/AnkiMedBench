# Sample Data Structure

This directory shows the expected format for your data files.

## ⚠️ Important

**AnkiMedBench does not include copyrighted Anki flashcard content.** You must obtain and prepare your own medical flashcard datasets.

## Expected Data Structure

```
data/
├── anki_cards/              # Your Anki flashcards (YOU MUST PROVIDE)
│   ├── front_text.txt      # One question per line
│   └── back_text.txt       # Corresponding answers
│
├── biosses/                 # BIOSSES dataset (publicly available)
│   └── BIOSSES.txt         # Download from official source
│
├── scitail/                 # SciTail dataset (HuggingFace)
│   ├── train/
│   ├── validation/
│   └── test/
│
└── pubmedqa/                # PubMedQA dataset (GitHub)
    ├── ori_pqaa.json
    └── ori_pqal.json
```

## Anki Cards Format

### front_text.txt
```
What is the mechanism of action of aspirin?
Which enzyme is deficient in phenylketonuria?
Describe the Frank-Starling mechanism.
```

### back_text.txt
```
Aspirin irreversibly inhibits cyclooxygenase (COX) enzymes, reducing prostaglandin synthesis and providing anti-inflammatory and antiplatelet effects.
Phenylalanine hydroxylase (PAH) is deficient in phenylketonuria, leading to accumulation of phenylalanine.
The Frank-Starling mechanism describes how increased venous return leads to increased stroke volume due to optimal sarcomere length and contractility.
```

## How to Obtain Medical Flashcards

### Option 1: Create Your Own
1. Install [Anki](https://apps.ankiweb.net/)
2. Create medical flashcards
3. Export: File → Export → Notes in Plain Text (.txt)

### Option 2: Use Licensed Decks (Check Terms)
- **AnKing**: Popular medical deck (verify license)
- **Zanki**: Step 1/2 focused deck (check usage rights)
- **Custom**: Your medical school's deck (with permission)

**Always verify licensing terms before using any deck!**

## Benchmark Datasets

All benchmark datasets are publicly available:

### BIOSSES
```bash
wget https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.zip
unzip DataSet.zip -d data/biosses/
```

### SciTail
```python
from datasets import load_dataset
dataset = load_dataset('scitail', 'snli_format')
dataset.save_to_disk('data/scitail/')
```

### PubMedQA
```bash
cd data/pubmedqa
wget https://github.com/pubmedqa/pubmedqa/raw/master/data/ori_pqaa.json
wget https://github.com/pubmedqa/pubmedqa/raw/master/data/ori_pqal.json
```

## Minimum Requirements

- **Anki Cards**: 1,000+ cards recommended (more is better)
- **Quality**: Medical accuracy is crucial
- **Format**: Plain text, UTF-8 encoding
- **Language**: English (or specify if different)

## Sample Size Recommendations

| Use Case | Minimum Cards | Recommended |
|----------|--------------|-------------|
| Testing | 100 | 500+ |
| Development | 1,000 | 5,000+ |
| Research | 5,000 | 10,000+ |
| Publication | 10,000+ | 20,000+ |

## Legal Notice

Users are responsible for ensuring they have appropriate rights to use any Anki flashcard content. AnkiMedBench framework itself is MIT licensed, but flashcard content may have separate licenses.
