from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .client import Refusal, chat_completion
from .dataset import IfStructExample, load_examples
from .validator import validate_response


@dataclass
class EvalResult:
    seed: int
    model: str
    passed: bool
    score: float
    errors: list[str]
    details: dict
    prompt: str
    response: str
    latency_ms: float
    output_format: str
    entity_type: str
    require_wrapper_key: bool


def run_one(example: IfStructExample, *, model: str, base_url: str, api_key: str, max_tokens: int, temperature: float, timeout: float, max_retries: int = 40) -> EvalResult:
    completion = chat_completion(
        base_url=base_url,
        api_key=api_key,
        model=model,
        prompt=example.prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
    )
    validation = validate_response(
        response=completion.text,
        json_schema=example.json_schema,
        top_level_count=example.top_level_count,
        require_no_commentary=example.require_no_commentary,
        output_format=example.output_format,
        top_level_key=example.top_level_key,
        require_wrapper_key=example.require_wrapper_key,
        require_code_block=example.require_code_block,
    )
    return EvalResult(
        seed=example.seed,
        model=model,
        passed=validation.passed,
        score=validation.score,
        errors=validation.errors,
        details=validation.details,
        prompt=example.prompt,
        response=completion.text,
        latency_ms=completion.latency_ms,
        output_format=example.output_format,
        entity_type=example.entity_type,
        require_wrapper_key=example.require_wrapper_key,
    )


def log_failure(path: str | None, result: EvalResult) -> None:
    if not path:
        return
    entry = {
        "timestamp": datetime.now().isoformat(),
        **asdict(result),
    }
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def build_summary(results: list[EvalResult]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    avg_latency = sum(result.latency_ms for result in results) / total if total else 0.0
    by_format: dict[str, dict[str, float | int]] = {}
    for output_format in ["json", "yaml"]:
        subset = [result for result in results if result.output_format == output_format]
        if subset:
            subset_passed = sum(1 for result in subset if result.passed)
            by_format[output_format] = {
                "passed": subset_passed,
                "total": len(subset),
                "pass_rate": subset_passed / len(subset),
            }

    by_top_level_structure: dict[str, dict[str, float | int]] = {}
    for label, flag in [("wrapper_key", True), ("bare_list", False)]:
        subset = [result for result in results if result.require_wrapper_key == flag]
        if subset:
            subset_passed = sum(1 for result in subset if result.passed)
            by_top_level_structure[label] = {
                "passed": subset_passed,
                "total": len(subset),
                "pass_rate": subset_passed / len(subset),
            }

    by_entity_type: dict[str, dict[str, float | int]] = {}
    for entity_type in sorted({result.entity_type for result in results}):
        subset = [result for result in results if result.entity_type == entity_type]
        subset_passed = sum(1 for result in subset if result.passed)
        by_entity_type[entity_type] = {
            "passed": subset_passed,
            "total": len(subset),
            "pass_rate": subset_passed / len(subset),
        }

    error_counts: Counter[str] = Counter()
    for result in results:
        if result.passed:
            continue
        for err in result.errors:
            if "required field missing" in err:
                key = "required field missing"
            elif "No valid JSON" in err:
                key = "no valid JSON"
            elif "No valid YAML" in err:
                key = "no valid YAML"
            elif "must use a code block" in err:
                key = "missing code block"
            elif "Expected wrapped object" in err:
                key = "expected wrapper key, got bare list"
            elif "Expected bare list" in err:
                key = "expected bare list, got wrapper"
            elif "Expected top-level key" in err:
                key = "wrong wrapper key name"
            elif "expected" in err and "got" in err:
                key = "type mismatch"
            elif "items" in err:
                key = "wrong item count"
            else:
                key = err[:80]
            error_counts[key] += 1

    return {
        "passed": passed,
        "total": total,
        "pass_rate": (passed / total) if total else 0.0,
        "average_latency_ms": avg_latency,
        "by_format": by_format,
        "by_top_level_structure": by_top_level_structure,
        "by_entity_type": by_entity_type,
        "common_errors": dict(error_counts.most_common(10)),
    }


def print_summary(results: list[EvalResult], model: str) -> None:
    summary = build_summary(results)

    print("\n" + "=" * 60)
    print(f"Model: {model}")
    print("=" * 60)
    print(f"Overall: {summary['passed']}/{summary['total']} passed ({100 * summary['pass_rate']:.1f}%)")
    print(f"Average latency: {summary['average_latency_ms']:.0f}ms")

    print("\nBy format:")
    for output_format in ["json", "yaml"]:
        subset = summary["by_format"].get(output_format)
        if subset:
            print(f"  {output_format.upper():4s}: {subset['passed']}/{subset['total']} passed ({100 * subset['pass_rate']:.1f}%)")

    print("\nBy top-level structure:")
    for label, key in [("Wrapper key", "wrapper_key"), ("Bare list", "bare_list")]:
        subset = summary["by_top_level_structure"].get(key)
        if subset:
            print(f"  {label:11s} {subset['passed']}/{subset['total']} passed ({100 * subset['pass_rate']:.1f}%)")

    print("\nBy entity type:")
    for entity_type, subset in summary["by_entity_type"].items():
        print(f"  {entity_type:35s} {subset['passed']}/{subset['total']} passed ({100 * subset['pass_rate']:.1f}%)")

    print("\nCommon errors:")
    for key, count in summary["common_errors"].items():
        print(f"  {count:3d}x {key}")


def has_api_failures(results: list[EvalResult]) -> bool:
    return any(bool(result.details.get("api_error")) for result in results)


def write_results_file(
    path: str,
    *,
    args: argparse.Namespace,
    started_at: datetime,
    finished_at: datetime,
    results: list[EvalResult],
) -> None:
    ordered_results = sorted(results, key=lambda result: result.seed)
    payload = {
        "run": {
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": (finished_at - started_at).total_seconds(),
            "model": args.model,
            "base_url": args.base_url,
            "dataset": args.dataset,
            "example_count": len(ordered_results),
            "n_threads": args.n_threads,
            "limit": args.limit,
            "offset": args.offset,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        },
        "summary": build_summary(ordered_results),
        "samples": [asdict(result) for result in ordered_results],
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run the standalone IFStruct eval.")
    parser.add_argument("--model", required=True, help="Model name to send to the OpenAI-compatible API.")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("BASE_URL"),
        help="Base URL for the OpenAI-compatible API. Defaults to $BASE_URL if set.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("API_KEY"),
        help="API key for the OpenAI-compatible API. Defaults to $API_KEY if set.",
    )
    parser.add_argument("--dataset", default="data/test.jsonl", help="Path to the precomputed dataset JSONL.")
    parser.add_argument("--n-threads", type=int, default=32, help="Parallel request workers.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on dataset rows.")
    parser.add_argument("--offset", type=int, default=0, help="Optional offset into the dataset.")
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=40, help="Max API retries per sample.")
    parser.add_argument("--seed", type=int, nargs="+", default=None, help="Run only these seed(s).")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--log-failures", default=None, help="Optional JSONL path for failed examples.")
    parser.add_argument("--results-file", default=None, help="Optional JSON path for run metadata, summary, and per-sample results.")
    args = parser.parse_args()

    if not args.base_url:
        parser.error("Missing --base-url or BASE_URL.")
    if not args.api_key:
        parser.error("Missing --api-key or API_KEY.")

    examples = load_examples(args.dataset)
    if args.seed:
        seed_set = set(args.seed)
        examples = [e for e in examples if e.seed in seed_set]
        if not examples:
            parser.error(f"No examples found for seed(s): {args.seed}")
    if args.offset:
        examples = examples[args.offset :]
    if args.limit is not None:
        examples = examples[: args.limit]

    started_at = datetime.now()

    print(f"Running IFStruct eval: {args.model}")
    print(f"Examples: {len(examples)}, Threads: {args.n_threads}")
    print()

    results: list[EvalResult] = []
    with ThreadPoolExecutor(max_workers=args.n_threads) as executor:
        futures = {
            executor.submit(
                run_one,
                example,
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
                max_retries=args.max_retries,
            ): example
            for example in examples
        }
        for future in as_completed(futures):
            example = futures[future]
            try:
                result = future.result()
            except Refusal as exc:
                result = EvalResult(
                    seed=example.seed,
                    model=args.model,
                    passed=False,
                    score=0.0,
                    errors=[f"Refusal: {exc}"],
                    details={"refusal": True},
                    prompt=example.prompt,
                    response="",
                    latency_ms=0.0,
                    output_format=example.output_format,
                    entity_type=example.entity_type,
                    require_wrapper_key=example.require_wrapper_key,
                )
            except Exception as exc:
                result = EvalResult(
                    seed=example.seed,
                    model=args.model,
                    passed=False,
                    score=0.0,
                    errors=[f"API error: {exc}"],
                    details={"api_error": True},
                    prompt=example.prompt,
                    response="",
                    latency_ms=0.0,
                    output_format=example.output_format,
                    entity_type=example.entity_type,
                    require_wrapper_key=example.require_wrapper_key,
                )
            results.append(result)
            if not result.passed:
                log_failure(args.log_failures, result)
            if args.verbose:
                status = "✓" if result.passed else "✗"
                print(f"  [{len(results):4d}/{len(examples)}] Seed {result.seed:4d}: {status} ({result.output_format}, {result.latency_ms:.0f}ms)")
                if not result.passed:
                    for err in result.errors[:2]:
                        print(f"      Error: {err[:120]}")

    finished_at = datetime.now()
    if args.results_file:
        write_results_file(
            args.results_file,
            args=args,
            started_at=started_at,
            finished_at=finished_at,
            results=results,
        )

    print_summary(results, args.model)
    if has_api_failures(results):
        print("\nRun completed with exhausted request retries. Failing the eval because one or more API requests never succeeded.")
        sys.exit(1)


if __name__ == "__main__":
    main()
