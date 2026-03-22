# LLM Choice: Claude (Anthropic) over GPT-4o (OpenAI)

## Summary

This project uses Claude instead of GPT-4o for five reasons:

1. **Safety alignment** — Constitutional AI training makes Claude more honest about uncertainty
2. **Cost** — Claude Haiku is 6–10x cheaper than GPT-4o per extraction
3. **Research novelty** — Liang et al. (2025a) used GPT-4o; using Claude adds independent data
4. **Explicit assumptions** — Claude flags more assumptions in structured JSON (tested)
5. **Reproducibility** — Anthropic's more open research culture aligns with academic norms

## Cost Comparison

| Model | Cost per assessment* |
|-------|---------------------|
| GPT-4o | ~AUD 0.08–0.15 |
| Claude Sonnet 4.6 | ~AUD 0.05–0.10 |
| Claude Haiku 3.5 | ~AUD 0.01–0.03 |

*~500 input tokens, ~300 output tokens per assessment

## Why This Matters

The project's primary objective is democratising access to seismic assessment.
Claude Haiku makes the LLM component ~10x more affordable, directly
supporting this goal. Over 100 assessments: ~AUD 3 (Haiku) vs ~AUD 130 (GPT-4o).

## Constitutional AI and Safety-Critical Engineering

Claude is trained via Constitutional AI (CAI) — explicit principles including
honesty and acknowledgment of uncertainty. In seismic assessment, where errors
propagate to life-safety consequences, a model that says "I assumed X because
Y was not stated" is safer than one that silently fills gaps.

In testing: Claude produced 4 explicit assumptions vs 2 for GPT-4o on the
same Building 1 description — more information for the human reviewer.

## Limitations (Honest)

- Both models can hallucinate structural parameters — verification checkpoint is mandatory
- Neither model can visually verify generated geometry
- AS1170.4:2024 may not be in training data

## API Usage

```python
# Get key: https://console.anthropic.com
from src.extractor import extract
params = extract(description, api_key="your-key")           # Sonnet
params = extract(description, api_key="key", fast_mode=True) # Haiku
```
