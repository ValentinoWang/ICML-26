# exp11 GPT-2 Streaming PPL Run (Absolute Path)

```bash
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python /root/autodl-tmp/ICML/STEP2/exp11_gpt2_model/run_exp11_gpt2_model.py \
  --device cuda \
  --save-checkpoints \
  --save-generations \
  --methods no_filter,pointwise,set_aware \
  --seeds 1088,2195,4960 \
  --results-path /root/autodl-tmp/ICML/STEP2/Total_results/Tables/exp11_gpt2_model/Results_streaming/metrics_diversity_ppl.json
```
