# FloodDepth-GPT baseline

FloodDepth-GPT estimates absolute flood depth from an image using a large multimodal model and a structured textual prompt. The baseline follows the prompt design described by Akinboyewa et al.:

- [Automated floodwater depth estimation using large multimodal model for rapid flood mapping](https://doi.org/10.1007/s43762-024-00123-3), *Computational Urban Science*, 4, 12 (2024).

The original method used GPT-4. As described in our paper, GPT-4 was not accessible at the time of our experiments, so we used GPT-4o as a more advanced substitute. The experiments were conducted manually through the ChatGPT chat interface rather than through the OpenAI API. Each evaluation image was submitted together with a textual prompt following the prompt design of the original FloodDepth-GPT study, and the returned text was saved for subsequent parsing and evaluation.

Because the experiment was performed through the ChatGPT interface, this repository does not provide an API inference script or initiate new model requests. Saved responses are provided to document the model outputs used in the paper and to make the result analysis reproducible.

## Saved results

The experimental outputs used by the paper analysis are stored with the other runtime results:

```text
runs/paper/baselines/flooddepth_gpt/
├── real_world/
│   ├── results.json
│   └── raw_responses.json
└── synthetic/
    ├── results.json
    └── raw_responses.json
```

For each dataset:

- `raw_responses.json` maps each evaluation image name directly to the text
  returned by GPT-4o, including the numerical answer and its explanation. The
  files contain no additional `raw_response` wrapper field.
- `results.json` contains the parsed per-image flood-depth estimates used in the paper comparison.

The response text was migrated from the original experiment notebook and
cross-checked against the numerical estimates in `results.json`. Both files
contain 200 entries for each evaluation dataset.

The paper result-analysis notebook reads the two `results.json` files directly. These saved files reproduce the reported analysis; submitting the same images again may produce different responses because model outputs are not deterministic.
