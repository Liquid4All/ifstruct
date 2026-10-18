from ifstruct.eval import EvalResult, build_summary


def _failed(error: str) -> EvalResult:
    return EvalResult(
        seed=0,
        model="m",
        passed=False,
        score=0.0,
        errors=[error],
        details={},
        prompt="p",
        response="r",
        latency_ms=0.0,
        output_format="json",
        entity_type="e",
        require_wrapper_key=False,
    )


def test_build_summary_buckets_parse_limit_errors():
    summary = build_summary(
        [
            _failed("JSON parse error: nesting exceeds the parser's recursion limit"),
            _failed("JSON parse error: number literal exceeds the 4300-digit int limit"),
        ]
    )

    assert summary["common_errors"] == {"nesting too deep": 1, "number literal too large": 1}
