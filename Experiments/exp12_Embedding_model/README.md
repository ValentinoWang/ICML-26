# Exp12: Embedding Topology (Manifold Volume)

Goal: show that Set-Aware preserves semantic manifold volume (no collapse) while maintaining quality.

## Dependencies
```bash
pip install sentence-transformers scikit-learn pandas seaborn matplotlib
```

## Data layout
Expected files (one folder per seed):
```
Experiments/Total_results/Tables/exp11_gpt2_model/generations/{seed}/g{gen}_{method}_{split}.txt
```
Example: `g4_pointwise_train.txt`, `g4_set_aware_train.txt`, `g0_no_filter_train.txt`.

Default settings use:
- generations: 0..4
- methods: `pointwise`, `set_aware`
- reference: `no_filter` (typically gen0)
- split: `train`

## Run analysis (Vendi + t-SNE)
```bash
python Experiments/exp12_Embedding_model/run_exp12_embedding_topology.py \
  --model Alibaba-NLP/gte-large-en-v1.5 \
  --data-root Experiments/Total_results/Tables/exp11_gpt2_model/generations \
  --out-root Experiments/Total_results
```

Use a different encoder if needed:
```bash
python Experiments/exp12_Embedding_model/run_exp12_embedding_topology.py \
  --model BAAI/bge-m3
```

Alias entrypoint (same behavior):
```bash
python Experiments/exp12_Embedding_model/calc_topology.py
```

## Refined plots
```bash
python Experiments/exp12_Embedding_model/plot_exp12_topology.py
```

Outputs:
- `Experiments/Total_results/Tables/exp12_Embedding_model/vendi_scores.csv`
- `Experiments/Total_results/Figures/exp12_Embedding_model/vendi_scores.png`
- `Experiments/Total_results/Tables/exp12_Embedding_model/tsne_g4_coords.csv`
- `Experiments/Total_results/Figures/exp12_Embedding_model/tsne_g4.png`
- refined plots: `vendi_scores_refined.png`, `tsne_g4_refined.png`
