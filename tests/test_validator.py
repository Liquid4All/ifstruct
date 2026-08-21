import pytest

from ifstruct.validator import (
    check_for_commentary,
    check_for_commentary_yaml,
    check_uses_code_block,
    extract_json_from_response,
    extract_yaml_from_response,
    remove_thinking_tags,
    validate_response,
)

# Captured from a real scoring run; cut off by the model's generation cap.
RUNAWAY_NUMBER_RESPONSE = (
    '{\n  "product_name": "NVIDIA GeForce RTX 4070 Ti Super",\n  "price_usd": ' + "24" + "9" * 8120
)


def test_unclosed_json_codeblock_is_not_extracted_as_raw_json():
    response = '```json\n[{"id": "1"}]'

    uses_block, block_type = check_uses_code_block(response, "json")
    data, error = extract_json_from_response(response)

    assert uses_block is False
    assert block_type is None
    assert data is None
    assert error == "Unclosed code block"


def test_unclosed_yaml_codeblock_is_not_extracted_as_raw_yaml():
    response = "```yaml\n- id: '1'"

    uses_block, block_type = check_uses_code_block(response, "yaml")
    data, error = extract_yaml_from_response(response)

    assert uses_block is False
    assert block_type is None
    assert data is None
    assert error == "Unclosed code block"


def test_validate_response_fails_unclosed_required_json_codeblock():
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
        "minItems": 1,
        "maxItems": 1,
    }
    response = '```json\n[{"id": "1"}]'

    result = validate_response(
        response=response,
        json_schema=schema,
        top_level_count=1,
        require_no_commentary=False,
        output_format="json",
        top_level_key=None,
        require_wrapper_key=False,
        require_code_block=True,
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.details["uses_code_block"] is False
    assert "Response must use a code block but none was found" in result.errors
    assert "Unclosed code block" in result.errors


def test_raw_json_and_yaml_still_parse_without_fences():
    json_data, json_error = extract_json_from_response('[{"id": "1"}]')
    yaml_data, yaml_error = extract_yaml_from_response("- id: '1'")

    assert json_error is None
    assert json_data == [{"id": "1"}]
    assert yaml_error is None
    assert yaml_data == [{"id": "1"}]


def test_raw_json_rejects_trailing_extra_brace():
    data, error = extract_json_from_response('{"id": "1"}\n}')

    assert data is None
    assert error is not None
    assert "Trailing content after JSON" in error


def test_raw_json_rejects_trailing_text():
    data, error = extract_json_from_response('[{"id": "1"}]\nok')

    assert data is None
    assert error is not None
    assert "Trailing content after JSON" in error


def test_json_codeblock_rejects_trailing_extra_brace_inside_fence():
    data, error = extract_json_from_response('```json\n{"id": "1"}\n}\n```')

    assert data is None
    assert error is not None
    assert "Trailing content after JSON" in error


def test_json_extraction_rejects_yaml_fence():
    data, error = extract_json_from_response('```yaml\n{"id": "1"}\n```')

    assert data is None
    assert error == "Expected JSON output, got YAML code block"


def test_yaml_extraction_rejects_json_fence():
    data, error = extract_yaml_from_response("```json\n- id: '1'\n```")

    assert data is None
    assert error == "Expected YAML output, got JSON code block"


def test_commentary_checks_flag_short_text_after_code_block():
    has_json_commentary, json_commentary = check_for_commentary(
        '```json\n[{"id": "1"}]\n```\nok'
    )
    has_yaml_commentary, yaml_commentary = check_for_commentary_yaml(
        "```yaml\n- id: '1'\n```\nok"
    )

    assert has_json_commentary is True
    assert json_commentary == 'Response contains text outside JSON: "ok"'
    assert has_yaml_commentary is True
    assert yaml_commentary == 'Response contains text outside YAML: "ok"'


def test_validate_response_fails_extra_brace_inside_json_codeblock():
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
        "minItems": 1,
        "maxItems": 1,
    }

    result = validate_response(
        response='```json\n[{"id": "1"}]\n}\n```',
        json_schema=schema,
        top_level_count=1,
        require_no_commentary=False,
        output_format="json",
        top_level_key=None,
        require_wrapper_key=False,
        require_code_block=True,
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.details["uses_code_block"] is True
    assert any("Trailing content after JSON" in error for error in result.errors)


def test_string_enum_mismatch_fails_validation():
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "unit": {"type": "string", "enum": ["cup", "tsp"]},
            },
            "required": ["unit"],
        },
        "minItems": 1,
        "maxItems": 1,
    }

    result = validate_response(
        response='[{"unit": "cups"}]',
        json_schema=schema,
        top_level_count=1,
        require_no_commentary=False,
        output_format="json",
        top_level_key=None,
        require_wrapper_key=False,
        require_code_block=False,
    )

    assert result.passed is False
    assert "'cups' not in allowed values ['cup', 'tsp']" in result.errors


def test_validate_response_strips_thinking_tags():
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
        "minItems": 1,
        "maxItems": 1,
    }

    result = validate_response(
        response='<think>draft invalid junk</think>[{"id": "1"}]',
        json_schema=schema,
        top_level_count=1,
        require_no_commentary=True,
        output_format="json",
        top_level_key=None,
        require_wrapper_key=False,
        require_code_block=False,
    )

    assert result.passed is True


def test_remove_thinking_tags_handles_harmony_final_channel():
    text = "analysis notes<|start|>assistant<|channel|>final<|message|>[{\"id\":\"1\"}]<|end|>"

    assert remove_thinking_tags(text) == '[{"id":"1"}]'


@pytest.mark.parametrize(
    "response",
    [RUNAWAY_NUMBER_RESPONSE, '```json\n{"price_usd": %s}\n```' % ("9" * 8122)],
    ids=["raw", "code_block"],
)
def test_json_extraction_rejects_oversized_number_literal(response):
    data, error = extract_json_from_response(response)

    assert data is None
    # The message must stay free of the per-response digit count.
    assert error == "JSON parse error: number literal exceeds the 4300-digit int limit"


def test_validate_response_fails_runaway_number_literal():
    schema = {
        "type": "object",
        "properties": {"product_name": {"type": "string"}, "price_usd": {"type": "number"}},
        "required": ["product_name", "price_usd"],
    }

    result = validate_response(
        response=RUNAWAY_NUMBER_RESPONSE,
        json_schema=schema,
        top_level_count=None,
        require_no_commentary=False,
        output_format="json",
        top_level_key=None,
        require_wrapper_key=False,
        require_code_block=False,
    )

    assert result.passed is False
    assert result.score == 0.0
    assert any("number literal exceeds" in error for error in result.errors)
