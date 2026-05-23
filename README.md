# 🔍 Transformer Architecture Deep Dive

Implement the full Transformer from scratch using PyTorch — scaled dot-product attention, multi-head attention, positional encoding, encoder-decoder stacks, and interactive attention-head visualizations.

## 🚀 Features
- Scaled dot-product & multi-head attention from scratch
- Sinusoidal and learned positional encodings
- Full encoder-decoder Transformer (Vaswani et al. 2017)
- Attention weight visualizations with matplotlib
- Training on WMT En→De translation task

## 📁 Structure
```
transformer-architecture-deep-dive/
├── src/
│   ├── attention.py        # Scaled dot-product & multi-head attention
│   ├── encoder.py          # Transformer encoder stack
│   ├── decoder.py          # Transformer decoder stack
│   ├── transformer.py      # Full model
│   └── visualize.py        # Attention head visualizations
├── notebooks/
│   └── 01_attention_walkthrough.ipynb
├── requirements.txt
└── README.md
```

## ⚡ Quick Start
```bash
pip install -r requirements.txt
python src/transformer.py --task translation --epochs 10
```

## 📊 Key Concepts Implemented
| Component | File |
|-----------|------|
| Scaled Dot-Product Attention | `src/attention.py` |
| Multi-Head Attention | `src/attention.py` |
| Positional Encoding | `src/encoder.py` |
| Feed-Forward Network | `src/encoder.py` |
| Full Transformer | `src/transformer.py` |
