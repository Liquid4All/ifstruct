from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import yaml


class _SafeLoaderNoDate(yaml.SafeLoader):
    pass


_SafeLoaderNoDate.yaml_implicit_resolvers = {
    k: [(tag, regexp) for tag, regexp in v if tag != "tag:yaml.org,2002:timestamp"]
    for k, v in yaml.SafeLoader.yaml_implicit_resolvers.copy().items()
}


@dataclass
class FieldCheck:
    path: str
    passed: bool
    error: str | None = None


@dataclass
class ValidationResult:
    passed: bool
    score: float
    errors: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def check_uses_code_block(response: str, output_format: str = "json") -> tuple[bool, str | None]:
    response = response.strip()
    _, opening_fence = _extract_outer_fenced_block(response)
    if opening_fence is not None:
        lowered = opening_fence.strip().lower()
        if lowered.startswith("```yaml"):
            return True, "```yaml"
        if lowered.startswith("```yml"):
            return True, "```yml"
        if lowered.startswith("```json"):
            return True, "```json"
        return True, "```"
    return False, None


def extract_json_from_response(response: str) -> tuple[Any | None, str | None]:
    response = response.strip()

    block, opening_fence = _extract_outer_fenced_block(response)
    if block is not None:
        lowered = (opening_fence or "").strip().lower()
        if not (lowered.startswith("```yaml") or lowered.startswith("```yml")):
            try:
                return json.loads(block.strip()), None
            except json.JSONDecodeError:
                pass

    for pattern in [r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"]:
        matches = re.findall(pattern, response)
        for match in matches:
            try:
                return json.loads(match.strip()), None
            except json.JSONDecodeError:
                continue

    first_brace = response.find("{")
    first_bracket = response.find("[")
    if first_brace == -1 and first_bracket == -1:
        return None, "No valid JSON found in response"

    if first_bracket == -1 or (first_brace != -1 and first_brace < first_bracket):
        char_order = [("{", "}"), ("[", "]")]
    else:
        char_order = [("[", "]"), ("{", "}")]

    for start_char, end_char in char_order:
        start_idx = response.find(start_char)
        if start_idx == -1:
            continue
        depth = 0
        in_string = False
        escape_next = False
        for i, c in enumerate(response[start_idx:], start_idx):
            if escape_next:
                escape_next = False
                continue
            if c == "\\":
                escape_next = True
                continue
            if c == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == start_char:
                depth += 1
            elif c == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(response[start_idx : i + 1]), None
                    except json.JSONDecodeError as exc:
                        return None, f"JSON parse error: {exc}"

    return None, "No valid JSON found in response"


def _format_yaml_error(exc: Exception) -> str:
    parts = []
    if hasattr(exc, "problem") and exc.problem:
        parts.append(exc.problem)
    if hasattr(exc, "problem_mark") and exc.problem_mark:
        mark = exc.problem_mark
        parts.append(f"at line {mark.line + 1}, column {mark.column + 1}")
        if hasattr(mark, "buffer") and mark.buffer:
            lines = mark.buffer.split("\n")
            if 0 <= mark.line < len(lines):
                problem_line = lines[mark.line].strip()
                if len(problem_line) > 60:
                    problem_line = problem_line[:60] + "..."
                parts.append(f"near: {problem_line!r}")
    return "; ".join(parts) if parts else str(exc)


def extract_yaml_from_response(response: str) -> tuple[Any | None, str | None]:
    response = response.strip()
    last_error: str | None = None
    block, opening_fence = _extract_outer_fenced_block(response)
    if block is not None:
        try:
            return yaml.load(block.strip(), Loader=_SafeLoaderNoDate), None
        except (yaml.YAMLError, ValueError) as exc:
            last_error = _format_yaml_error(exc)
    for pattern in [r"```yaml\s*([\s\S]*?)\s*```", r"```yml\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"]:
        matches = re.findall(pattern, response)
        for match in matches:
            try:
                return yaml.load(match.strip(), Loader=_SafeLoaderNoDate), None
            except (yaml.YAMLError, ValueError) as exc:
                last_error = _format_yaml_error(exc)
    try:
        return yaml.load(response, Loader=_SafeLoaderNoDate), None
    except (yaml.YAMLError, ValueError) as exc:
        last_error = _format_yaml_error(exc)
    if last_error:
        return None, f"YAML parsing error: {last_error}"
    return None, "No valid YAML found in response (YAML must be in a ```yaml code block)"


def _extract_outer_fenced_block(response: str) -> tuple[str | None, str | None]:
    lines = response.strip().splitlines()
    start_idx = None
    opening_fence = None

    for i, line in enumerate(lines):
        if line.startswith("```"):
            start_idx = i
            opening_fence = line
            break

    if start_idx is None:
        return None, None

    for j in range(start_idx + 1, len(lines)):
        if lines[j].startswith("```"):
            return "\n".join(lines[start_idx + 1 : j]), opening_fence

    return None, opening_fence


def _find_json_boundaries(response: str) -> tuple[int, int] | None:
    first_brace = response.find("{")
    first_bracket = response.find("[")
    if first_brace == -1 and first_bracket == -1:
        return None

    if first_bracket == -1 or (first_brace != -1 and first_brace < first_bracket):
        char_order = [("{", "}"), ("[", "]")]
    else:
        char_order = [("[", "]"), ("{", "}")]

    for start_char, end_char in char_order:
        start_idx = response.find(start_char)
        if start_idx == -1:
            continue
        depth = 0
        in_string = False
        escape_next = False
        for i, c in enumerate(response[start_idx:], start_idx):
            if escape_next:
                escape_next = False
                continue
            if c == "\\":
                escape_next = True
                continue
            if c == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == start_char:
                depth += 1
            elif c == end_char:
                depth -= 1
                if depth == 0:
                    return start_idx, i
    return None


def check_for_commentary(response: str) -> tuple[bool, str | None]:
    response = response.strip()
    block, _ = _extract_outer_fenced_block(response)
    if block is not None:
        before = response[: response.find("```")].strip()
        end_marker = response.rfind("```")
        after = response[end_marker + 3 :].strip() if end_marker != -1 else ""
        remaining = f"{before} {after}".strip()
    else:
        boundaries = _find_json_boundaries(response)
        if boundaries is None:
            return False, None
        start_idx, end_idx = boundaries
        before = response[:start_idx].strip()
        after = response[end_idx + 1 :].strip()
        remaining = f"{before} {after}".strip()

    remaining = remaining.strip()
    if not remaining:
        return False, None
    if len(remaining) > 10:
        preview = remaining[:100] + "..." if len(remaining) > 100 else remaining
        return True, f'Response contains text outside JSON: "{preview}"'
    return False, None


def check_for_commentary_yaml(response: str) -> tuple[bool, str | None]:
    response = response.strip()
    block, _ = _extract_outer_fenced_block(response)
    if block is not None:
        before = response[: response.find("```")].strip()
        end_marker = response.rfind("```")
        after = response[end_marker + 3 :].strip() if end_marker != -1 else ""
        remaining = f"{before} {after}".strip()
    else:
        return False, None
    remaining = remaining.strip()
    if not remaining:
        return False, None
    if len(remaining) > 10:
        preview = remaining[:100] + "..." if len(remaining) > 100 else remaining
        return True, f'Response contains text outside YAML: "{preview}"'
    return False, None


def validate_against_json_schema(data: Any, schema: dict[str, Any], path: str = "") -> list[FieldCheck]:
    checks: list[FieldCheck] = []
    schema_type = schema.get("type")

    if isinstance(schema_type, list):
        if data is None:
            if "null" in schema_type:
                return [FieldCheck(path or "root", True)]
            return [FieldCheck(path or "root", False, f"expected {schema_type}, got NoneType")]
        non_null_types = [t for t in schema_type if t != "null"]
        if len(non_null_types) == 1:
            schema_type = non_null_types[0]

    if data is None and schema_type != "null":
        return [FieldCheck(path or "root", False, f"expected {schema_type}, got NoneType")]

    if schema_type == "array":
        if not isinstance(data, list):
            return [FieldCheck(path or "root", False, f"expected array, got {type(data).__name__}")]
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if min_items is not None and len(data) < min_items:
            checks.append(FieldCheck(path or "root", False, f"array has {len(data)} items, minimum is {min_items}"))
        elif max_items is not None and len(data) > max_items:
            checks.append(FieldCheck(path or "root", False, f"array has {len(data)} items, maximum is {max_items}"))
        items_schema = schema.get("items")
        if items_schema:
            for i, item in enumerate(data):
                item_path = f"{path}[{i}]" if path else f"[{i}]"
                checks.extend(validate_against_json_schema(item, items_schema, item_path))
        return checks

    if schema_type == "object":
        if not isinstance(data, dict):
            return [FieldCheck(path or "root", False, f"expected object, got {type(data).__name__}")]
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        for field_name in required:
            field_path = f"{path}.{field_name}" if path else field_name
            if field_name not in data:
                leaf_count = _count_schema_leaves(properties[field_name]) if field_name in properties else 1
                for _ in range(leaf_count):
                    checks.append(FieldCheck(field_path, False, "required field missing"))
        for field_name, field_schema in properties.items():
            if field_name in data:
                field_path = f"{path}.{field_name}" if path else field_name
                checks.extend(validate_against_json_schema(data[field_name], field_schema, field_path))
        extra_keys = set(data.keys()) - set(properties.keys())
        for key in sorted(extra_keys, key=str):
            field_path = f"{path}.{str(key)}" if path else str(key)
            checks.append(FieldCheck(field_path, False, f"extraneous field '{key}'"))
        return checks

    if schema_type == "string":
        return [FieldCheck(path or "root", isinstance(data, str), None if isinstance(data, str) else f"expected string, got {type(data).__name__}")]

    if schema_type == "number":
        if not isinstance(data, (int, float)) or isinstance(data, bool):
            return [FieldCheck(path or "root", False, f"expected number, got {type(data).__name__}")]
        error = None
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and data < minimum:
            error = f"{data} is less than minimum {minimum}"
        elif maximum is not None and data > maximum:
            error = f"{data} is greater than maximum {maximum}"
        checks.append(FieldCheck(path or "root", error is None, error))
    elif schema_type == "integer":
        if not isinstance(data, int) or isinstance(data, bool):
            return [FieldCheck(path or "root", False, f"expected integer, got {type(data).__name__}")]
        error = None
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and data < minimum:
            error = f"{data} is less than minimum {minimum}"
        elif maximum is not None and data > maximum:
            error = f"{data} is greater than maximum {maximum}"
        checks.append(FieldCheck(path or "root", error is None, error))
    elif schema_type == "boolean":
        checks.append(FieldCheck(path or "root", isinstance(data, bool), None if isinstance(data, bool) else f"expected boolean, got {type(data).__name__}"))

    enum_values = schema.get("enum")
    if enum_values is not None:
        checks.append(FieldCheck(path or "root", data in enum_values, None if data in enum_values else f"{data!r} not in allowed values {enum_values}"))
    return checks


def _count_schema_leaves(schema: dict[str, Any]) -> int:
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        non_null = [t for t in schema_type if t != "null"]
        schema_type = non_null[0] if len(non_null) == 1 else schema_type
    if schema_type == "object":
        properties = schema.get("properties", {})
        if not properties:
            return 1
        required = schema.get("required")
        if required is None:
            selected = list(properties.values())
        else:
            selected = [properties[name] for name in required if name in properties]
        return sum(_count_schema_leaves(s) for s in selected) if selected else 1
    if schema_type == "array":
        items_schema = schema.get("items")
        return _count_schema_leaves(items_schema) if items_schema else 1
    count = 1
    if schema.get("enum") is not None:
        count += 1
    return count


def _compute_expected_checks(json_schema: dict[str, Any], top_level_count: int | list[int] | None) -> int:
    if json_schema.get("type") == "array":
        items_schema = json_schema.get("items")
        if not items_schema:
            return 0
        leaves_per_item = _count_schema_leaves(items_schema)
        if isinstance(top_level_count, int):
            return top_level_count * leaves_per_item
        if isinstance(top_level_count, list) and len(top_level_count) == 2:
            return top_level_count[0] * leaves_per_item
        return 0
    return _count_schema_leaves(json_schema)


def _check_top_level_structure(data: Any, expected_key: str | None, require_wrapper: bool) -> tuple[Any, bool, str | None]:
    if isinstance(data, list):
        if require_wrapper:
            return data, False, f"Expected wrapped object with key '{expected_key}', got bare list"
        return data, False, None
    if isinstance(data, dict) and len(data) == 1:
        only_key = next(iter(data.keys()))
        only_value = data[only_key]
        if isinstance(only_value, list):
            if not require_wrapper:
                return only_value, True, f"Expected bare list, got wrapped object with key '{only_key}'"
            if expected_key is None or only_key == expected_key:
                return only_value, True, None
            return only_value, True, f"Expected top-level key '{expected_key}', got '{only_key}'"
    return data, False, None


def validate_response(
    *,
    response: str,
    json_schema: dict[str, Any],
    top_level_count: int | list[int] | None,
    require_no_commentary: bool,
    output_format: str,
    top_level_key: str | None,
    require_wrapper_key: bool,
    require_code_block: bool,
) -> ValidationResult:
    errors: list[str] = []
    details: dict[str, Any] = {"output_format": output_format}

    uses_code_block, code_block_type = check_uses_code_block(response, output_format)
    details["uses_code_block"] = uses_code_block
    details["code_block_type"] = code_block_type
    if require_code_block and not uses_code_block:
        errors.append("Response must use a code block but none was found")

    if output_format == "yaml":
        parsed, extract_error = extract_yaml_from_response(response)
        details["yaml_valid"] = extract_error is None
    else:
        parsed, extract_error = extract_json_from_response(response)
        details["json_valid"] = extract_error is None
    if extract_error:
        return ValidationResult(False, 0.0, errors + [extract_error], details)

    if require_no_commentary:
        if output_format == "yaml":
            has_commentary, commentary_desc = check_for_commentary_yaml(response)
        else:
            has_commentary, commentary_desc = check_for_commentary(response)
        details["no_commentary"] = not has_commentary
        if has_commentary and commentary_desc:
            errors.append(commentary_desc)

    parsed, was_wrapped, wrap_error = _check_top_level_structure(parsed, top_level_key, require_wrapper_key)
    details["was_wrapped"] = was_wrapped
    if wrap_error:
        errors.append(wrap_error)

    field_checks = validate_against_json_schema(parsed, json_schema)
    schema_errors = [check.error for check in field_checks if not check.passed and check.error]
    if schema_errors:
        errors.extend(schema_errors)
        details["schema_valid"] = False
    else:
        details["schema_valid"] = True

    if top_level_count is not None and isinstance(parsed, list):
        actual_count = len(parsed)
        if isinstance(top_level_count, int):
            if actual_count != top_level_count:
                errors.append(f"Expected {top_level_count} items, got {actual_count}")
        elif isinstance(top_level_count, list) and len(top_level_count) == 2:
            min_count, max_count = top_level_count
            if actual_count < min_count or actual_count > max_count:
                errors.append(f"Expected {min_count}-{max_count} items, got {actual_count}")

    passed_checks = sum(1 for check in field_checks if check.passed)
    total_checks = max(_compute_expected_checks(json_schema, top_level_count), len(field_checks))
    details["schema_fields_total"] = total_checks
    details["schema_fields_passed"] = passed_checks
    details["schema_match_ratio"] = passed_checks / total_checks if total_checks else 0.0
    passed = len(errors) == 0
    return ValidationResult(passed=passed, score=1.0 if passed else 0.0, errors=errors, details=details)
