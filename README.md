# Tea

The Technical Ethics Analyzer is a software solution developed to assess the ethical reasoning of large language models (LLMs) from multiple providers by submitting a vast array of ethical prompts for quantitative analysis.

## Functions/Features

### Prompting
Teas primary purpose is to prompt at scale, which is provided as a python app in the prompting folder.

You can run it via:
You can run it via:
```bash
make prompting
```
Or manually:
```bash
uv run --directory prompting main.py
```

### Analysing
As we are analysing with a specific prompt template Tea also provides some basic analytic Tools and a UI made with Nicegui.

You can run it via:
```bash
make analysing
```
Or manually:
```bash
uv run --directory analysing main.py
```

## Utility

### Package Manager
The Tea uses uv as a python package manager.
You can find full documentation [here](https://docs.astral.sh/uv/).