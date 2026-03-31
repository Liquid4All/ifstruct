# IFStruct

IFStruct is an eval for instruction-following on structured output tasks.

It tests whether a model can produce JSON or YAML that matches a target schema while also following formatting constraints such as:

- returning the right top-level shape
- using the required wrapper key when one is specified
- emitting a code block when requested
- avoiding extra commentary outside the structured output

This repo includes the dataset, validator, and a small CLI runner for evaluating models against the benchmark.

## Install

```bash
cd ~/code/evals/ifstruct
uv sync
```

## Configure

Create a local env file from the example:

```bash
cp .env.example .env
```

Set these values in `.env`:

```bash
BASE_URL=https://openrouter.ai/api/v1
API_KEY=your-api-key-here
```

The CLI automatically loads `.env` if it exists, and it also works if `BASE_URL` and `API_KEY` are already exported in your shell.

## Run

```bash
uv run ifstruct-eval \
  --model google/gemini-3-flash-preview \
  --dataset data/test.jsonl \
  --results-file results/latest.json \
  --n-threads 64 \
  -v
```

For a small smoke test:

```bash
uv run ifstruct-eval \
  --model google/gemini-3-flash-preview \
  --dataset data/test.jsonl \
  --limit 20 \
  --results-file results/smoke.json \
  --n-threads 8 \
  -v
```

You can still override either setting explicitly with `--base-url` or `--api-key`.
`--results-file` writes a JSON artifact with run metadata, aggregate summary stats, and per-sample prompts, responses, and validation results.

## Methodology

IFStruct uses a 2,000-example test set in [data/test.jsonl](/Users/sam/code/evals/ifstruct/data/test.jsonl). Each example includes a prompt, a target schema, and explicit structural requirements for the response.

The eval sends each prompt to an OpenAI-compatible chat completions endpoint, captures the model response, and validates it against:

- the requested output format: JSON or YAML
- the expected top-level structure: bare list or wrapped object
- the required wrapper key when applicable
- code-block requirements
- no-commentary requirements
- the provided schema for item-level fields and types

The CLI prints aggregate pass rates and can also write a full results artifact containing run metadata plus every prompt, response, and validation result.

## Dataset Format

Each JSONL row contains:

- `seed`
- `entity_type`
- `prompt`
- `json_schema`
- `top_level_count`
- `top_level_key`
- `require_wrapper_key`
- `require_code_block`
- `require_no_commentary`
- `output_format`
