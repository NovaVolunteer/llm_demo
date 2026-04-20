# llm_demo
Quick Overview of LLMs

## Temperature tuning demo

Run this script to compare generation at low and high temperature with a small free Hugging Face model.

```bash
pip install transformers torch
python /home/runner/work/llm_demo/llm_demo/temperature_tuning_demo.py
```

The output includes a simple repetition score so you can see how very low temperature often makes text more repetitive.
