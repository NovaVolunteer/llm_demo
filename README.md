# llm_demo

Quick Overview of LLMs

## Temperature tuning demo

Run this script to compare generation at low and high temperature with a small free Hugging Face model.

```bash
pip install transformers torch
python temperature_tuning_demo.py
```

Defaults are tuned for a clearer comparison:

- model: distilgpt2
- max_new_tokens: 120
- separate seeds for low/high runs (high defaults to seed + 1)

The output includes a simple repetition score so you can see how very low temperature often makes text more repetitive.
