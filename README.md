# IFStruct

IFStruct is an eval for instruction-following on structured output tasks.

It tests whether a model can produce valid JSON or YAML that matches a target schema:

"Produce a recipe for blueberry muffins in JSON with this schema..."

"Generate five entries in a code review log for a llama.cpp PR, in valid YAML following this schema..."

## Why does this exist?

Frontier models can produce more or less perfect JSON and YAML to a schema, but small models still struggle, producing unparseable outputs and ignoring schema constraints. This is a very learnable task for even tiny (350M) models, however there are few evals targeting structured output and its common failure modes.

## IFStruct Difficulty

The difficulty of IFStruct is calibrated so that frontier models are expected to score near 100%. The tasks contain challenges such as:

- Nested structures
- Strings requiring escaped chars
- Constraints on data type, value ranges, enums and number of items

## Verification and Scoring

Each response is run through a multi-stage verification pipeline:

1. **Parse check** — can the response be parsed as valid JSON or YAML?
2. **Code-block check** — if the prompt required a fenced code block, is one present?
3. **No-commentary check** — if the prompt required no extra text, is the response free of commentary outside the structured output?
4. **Structure check** — does the top-level shape match what was asked for (bare list `[...]` vs. wrapped object `{"key": [...]}`), and does the wrapper key name match?
5. **Item-count check** — does the number of items match the required count or range?
6. **Schema check** — every leaf field in every item is validated against the provided JSON schema for correct types, required fields, enum membership, and numeric min/max bounds. Extraneous fields are also flagged.

A sample **passes** only if all checks produce zero errors (score = 1). Any error means a fail (score = 0). The overall score is the pass rate across the test set.

The results file also reports pass rates sliced by output format (JSON vs. YAML), top-level structure (bare list vs. wrapper key), and entity type, along with the ten most common error categories.

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
