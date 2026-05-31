import argparse
import csv
import importlib.util
import json
import os
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


DEFAULT_CACHE_DIR = "/data/SUYUKANG/huggingface/hub/"
DEFAULT_MODEL = os.environ.get("QWEN3_VL_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
# The revised prompt asks Phase-2 relevance and Phase-3 impact for six
# context dimensions, including Persona. Keep this order identical to the
# prompt's Output Requirements.
EXPECTED_KEYS = [
    "RA-1", "IA-1",
    "RA-2", "IA-2",
    "RA-3", "IA-3",
    "RA-4", "IA-4",
    "RA-5", "IA-5",
    "RA-6", "IA-6",
]

RELEVANCE_LABELS = {"Highly relevant", "Relevant", "Not relevant"}
IMPACT_LABELS = {"Significant", "Marginal", "Negligible"}

CONTEXT_KEYS = [
    "Agent-specific context",
    "Object-specific context",
    "Environmental context",
    "Cultural context",
    "Temporal context",
]

CONTEXT_KEY_MAP = {
    "<Agent-specific context>": "Agent-specific context",
    "<Object-specific context>": "Object-specific context",
    "<Environmental context>": "Environmental context",
    "<Cultural context>": "Cultural context",
    "<Temporal context>": "Temporal context",
}

PERSONA_KEYS = [
    "Demographics",
    "Physical attributes",
    "Personality",
    "Emotional state",
    "Social roles",
    "Hobbies",
    "Cultural characteristics",
]


def load_python_module(module_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_prompt_template(prompt_py: str) -> str:
    module = load_python_module(prompt_py, "prompt_module")
    if not hasattr(module, "prompt"):
        raise ValueError(f"No variable named 'prompt' found in {prompt_py}")
    return str(module.prompt)


def load_json(json_path: str) -> Any:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_rgb(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def resolve_image_path(images_dir: str, image_id: str) -> str:
    path = os.path.join(images_dir, image_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return path


def stringify_compact(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def format_persona_for_prompt(persona: Any) -> str:
    if persona is None:
        return ""
    if isinstance(persona, str):
        return persona
    if isinstance(persona, dict):
        ordered_items = []
        for key in PERSONA_KEYS:
            if key in persona:
                ordered_items.append(f"{key}: {stringify_compact(persona.get(key, ''))}")

        for key, value in persona.items():
            if key not in PERSONA_KEYS:
                ordered_items.append(f"{key}: {stringify_compact(value)}")

        return "; ".join(ordered_items)
    return stringify_compact(persona)


def normalize_affordance_item(affordance_item: Dict[str, Any]) -> Dict[str, Any]:
    context = _get_context_dict(affordance_item)

    # Support both layouts:
    # 1) Persona stored as a top-level field in each affordance item.
    # 2) Persona stored inside Context, matching the revised prompt input block.
    persona = affordance_item.get("Persona", context.get("Persona", {}))

    return {
        "Agent": str(affordance_item.get("Agent", "")),
        "Object": str(affordance_item.get("Object", "")),
        "Action": str(affordance_item.get("Action", "")),
        "Context": {key: stringify_compact(context.get(key, "")) for key in CONTEXT_KEYS},
        "Persona": persona,
    }


def _require_core_field(data: Dict[str, Any], field_name: str) -> Any:
    if field_name not in data:
        raise KeyError(f"Missing required field: {field_name}")
    return data[field_name]


def _get_context_dict(affordance_item: Dict[str, Any]) -> Dict[str, Any]:
    context = affordance_item.get("Context", {})
    if not isinstance(context, dict):
        raise ValueError(f"Context must be a dict, got: {type(context).__name__}")
    return context


def _get_persona_value(affordance_item: Dict[str, Any]) -> Any:
    context = _get_context_dict(affordance_item)
    return affordance_item.get("Persona", context.get("Persona", {}))


def build_prompt(prompt_template: str, affordance_item: Dict[str, Any]) -> str:
    prompt = str(prompt_template)
    normalized = normalize_affordance_item(affordance_item)

    replacements = {
        "<Agent>": str(_require_core_field(affordance_item, "Agent")),
        "<Object>": str(_require_core_field(affordance_item, "Object")),
        "<Objet>": str(_require_core_field(affordance_item, "Object")),  # backward-compatible typo
        "<Action>": str(_require_core_field(affordance_item, "Action")),
        "<Persona>": format_persona_for_prompt(normalized["Persona"]),
    }

    for placeholder, context_key in CONTEXT_KEY_MAP.items():
        replacements[placeholder] = normalized["Context"].get(context_key, "")

    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, stringify_compact(value))

    return prompt

def load_model_and_processor(
    model_name: str,
    device_map: str,
    dtype: str,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    use_flash_attention_2: bool = False,
):
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}

    model_kwargs = {
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
        "device_map": device_map,
    }

    if use_flash_attention_2:
        try:
            import flash_attn  # noqa: F401
            model_kwargs["dtype"] = dtype_map[dtype]
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("FlashAttention2 enabled.")
        except ImportError:
            print("Warning: flash_attn is not installed. Falling back to default attention.")
            model_kwargs["dtype"] = "auto"
    else:
        model_kwargs["dtype"] = "auto"

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        **model_kwargs,
    ).eval()

    processor = AutoProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    return model, processor


def run_single_inference(
    model,
    processor,
    image_path: str,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[str, Dict[str, Any]]:
    image = ensure_rgb(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
        "use_cache": True,
    }
    if temperature > 0.0:
        generation_config["temperature"] = temperature
        generation_config["top_p"] = top_p

    with torch.inference_mode():
        gen_ids = model.generate(**inputs, **generation_config)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return response, generation_config


def _strip_code_fence(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.DOTALL).strip()


def _extract_json_candidate(text: str) -> str:
    cleaned = _strip_code_fence(text)
    if not cleaned:
        return cleaned

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start:end + 1]
    return cleaned


def format_evaluator_value(value: Any) -> Any:
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        normalized = list(value)
        if len(normalized) == 3:
            confidence = normalized[1]
            if isinstance(confidence, str):
                try:
                    normalized[1] = float(confidence)
                except ValueError:
                    pass
        return normalized
    if value is None:
        return ""
    return value


def _expected_labels_for_key(key: str) -> set:
    if key.startswith("RA-"):
        return RELEVANCE_LABELS
    if key.startswith("IA-"):
        return IMPACT_LABELS
    return set()


def _validate_evaluator_output(parsed: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    parsed_keys = list(parsed.keys())
    missing_keys = [key for key in EXPECTED_KEYS if key not in parsed]
    extra_keys = [key for key in parsed_keys if key not in EXPECTED_KEYS]
    if missing_keys:
        errors.append(f"Missing keys: {missing_keys}")
    if extra_keys:
        errors.append(f"Unexpected keys: {extra_keys}")
    if parsed_keys != EXPECTED_KEYS:
        errors.append("Output key order does not match the revised prompt requirements.")

    for key in EXPECTED_KEYS:
        value = parsed.get(key)
        if not isinstance(value, list) or len(value) != 3:
            errors.append(f"{key} must be a 3-element array [Label, Confidence, Explanation].")
            continue

        label, confidence, explanation = value
        if label not in _expected_labels_for_key(key):
            errors.append(f"{key} label is invalid: {label!r}")

        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            errors.append(f"{key} confidence must be a number from 0.00 to 1.00.")
        else:
            if not 0.0 <= confidence_value <= 1.0:
                errors.append(f"{key} confidence must be a number from 0.00 to 1.00.")

        if not isinstance(explanation, str):
            errors.append(f"{key} explanation must be a string.")

    return errors


def parse_vlm_json_output(text: str) -> Tuple[Dict[str, Any], Optional[str]]:
    raw_candidate = _extract_json_candidate(text)
    parse_errors: List[str] = []
    parsed: Dict[str, Any] = {}

    try:
        loaded = json.loads(raw_candidate)
        if isinstance(loaded, dict):
            parsed = loaded
            parse_errors.extend(_validate_evaluator_output(parsed))
        else:
            parse_errors.append("Model output is not a JSON object.")
    except Exception as e:
        parse_errors.append(f"Failed to parse model output as JSON: {e}")

    normalized: Dict[str, Any] = OrderedDict()
    for key in EXPECTED_KEYS:
        normalized[key] = format_evaluator_value(parsed.get(key, ""))

    parse_error = "; ".join(parse_errors) if parse_errors else None
    return normalized, parse_error

def read_existing_results(output_path: str) -> Dict[str, Any]:
    if not os.path.exists(output_path):
        return {}
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def remove_raw_output_fields(data: Any) -> Any:
    if isinstance(data, dict):
        data.pop("Raw_Output", None)
        data.pop("raw_output", None)
        for value in data.values():
            remove_raw_output_fields(value)
    elif isinstance(data, list):
        for item in data:
            remove_raw_output_fields(item)
    return data


def save_results(output_path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    data = remove_raw_output_fields(data)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_default_error_csv_path(output_json_path: str) -> str:
    base_name = os.path.splitext(os.path.basename(output_json_path))[0]
    return os.path.join("Log", f"{base_name}_errors.csv")


def ensure_error_csv_header(error_csv_path: str) -> None:
    os.makedirs(os.path.dirname(error_csv_path) or ".", exist_ok=True)
    if os.path.exists(error_csv_path) and os.path.getsize(error_csv_path) > 0:
        return

    with open(error_csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_id",
            "affordance_index",
            "agent",
            "object",
            "action",
            "error_type",
            "error_message",
            "raw_output",
        ])


def append_error_record(
    error_csv_path: str,
    image_id: str,
    affordance_index: int,
    affordance_item: Optional[Dict[str, Any]],
    error_type: str,
    error_message: str,
    raw_output: str,
) -> None:
    ensure_error_csv_header(error_csv_path)

    agent = str(affordance_item.get("Agent", "")) if affordance_item else ""
    obj = str(affordance_item.get("Object", "")) if affordance_item else ""
    action = str(affordance_item.get("Action", "")) if affordance_item else ""

    with open(error_csv_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            image_id,
            affordance_index,
            agent,
            obj,
            action,
            error_type,
            error_message,
            raw_output,
        ])


def init_result_slot(results: Dict[str, Any], image_id: str, affordance_index: int, affordance_item: Dict[str, Any]) -> None:
    normalized_item = normalize_affordance_item(affordance_item)
    image_bucket = results.setdefault(image_id, {})
    result_bucket = image_bucket.setdefault(str(affordance_index), {})
    result_bucket["Core_Triplet"] = {
        "Agent": normalized_item["Agent"],
        "Object": normalized_item["Object"],
        "Action": normalized_item["Action"],
    }
    result_bucket["Context"] = {
        **normalized_item["Context"],
        "Persona": normalized_item["Persona"],
    }
    # Remove legacy fields if this output JSON was created by an older script.
    result_bucket.pop("Affordance", None)
    result_bucket.pop("Persona", None)

def save_evaluation_result(
    output_path: str,
    results: Dict[str, Any],
    image_id: str,
    affordance_index: int,
    affordance_item: Dict[str, Any],
    evaluator_output: Dict[str, Any],
) -> None:
    init_result_slot(results, image_id, affordance_index, affordance_item)
    bucket = results[image_id][str(affordance_index)]
    bucket["Evaluator_Output"] = evaluator_output
    bucket.pop("Raw_Output", None)
    bucket.pop("raw_output", None)
    save_results(output_path, results)


def main():
    parser = argparse.ArgumentParser(description="Evaluate revised affordance context/persona relevance and impact with Qwen3-VL.")
    parser.add_argument("--images_dir", type=str, default="Data/Framework_Data/Golden Label-image-500", help="Directory containing all images.")
    parser.add_argument("--prompt_py", type=str, default="Data/Framework_Data/prompt.py")
    parser.add_argument(
        "--input_json", "--triplet_json",
        dest="input_json",
        type=str,
        default="Data/Evaluation_Data/Affordance-centric_knowledge_GPT4o_verified_500.json",
    )
    parser.add_argument("--output_json", type=str, default="Output/output_context_Qwen3-VL(8B)_GPT4o_verified.json")
    parser.add_argument("--error_csv", type=str, default=None)
    parser.add_argument("--model", "--model_name", dest="model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--use_flash_attention_2", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit_images", type=int, default=None)
    args = parser.parse_args()

    error_csv_path = args.error_csv or get_default_error_csv_path(args.output_json)

    if args.overwrite and os.path.exists(args.output_json):
        os.remove(args.output_json)
    if args.overwrite and os.path.exists(error_csv_path):
        os.remove(error_csv_path)

    prompt_template = load_prompt_template(args.prompt_py)
    evaluation_data = load_json(args.input_json)

    if args.limit_images is not None:
        evaluation_data = evaluation_data[: args.limit_images]

    model, processor = load_model_and_processor(
        model_name=args.model,
        device_map=args.device_map,
        dtype=args.dtype,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        use_flash_attention_2=args.use_flash_attention_2,
    )

    results = {} if args.overwrite else remove_raw_output_fields(read_existing_results(args.output_json))

    for image_entry in evaluation_data:
        image_id = image_entry["Image_id"]
        affordances = image_entry.get("Affordance", [])

        try:
            image_path = resolve_image_path(args.images_dir, image_id)
        except Exception as e:
            append_error_record(
                error_csv_path=error_csv_path,
                image_id=image_id,
                affordance_index=-1,
                affordance_item=None,
                error_type="image_error",
                error_message=str(e),
                raw_output="",
            )
            print(f"[IMAGE_ERROR] image={image_id} error={e}")
            continue

        for affordance_index, affordance_item in enumerate(affordances):
            init_result_slot(results, image_id, affordance_index, affordance_item)

            try:
                prompt_text = build_prompt(prompt_template, affordance_item)
                raw_output, generation_config = run_single_inference(
                    model=model,
                    processor=processor,
                    image_path=image_path,
                    prompt_text=prompt_text,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                evaluator_output, parse_error = parse_vlm_json_output(raw_output)

                save_evaluation_result(
                    output_path=args.output_json,
                    results=results,
                    image_id=image_id,
                    affordance_index=affordance_index,
                    affordance_item=affordance_item,
                    evaluator_output=evaluator_output,
                )

                status = "OK" if parse_error is None else "CHECK"
                print(
                    f"[{status}] image={image_id} affordance={affordance_index} "
                    f"parsed={json.dumps(evaluator_output, ensure_ascii=False)}"
                )

                if parse_error:
                    append_error_record(
                        error_csv_path=error_csv_path,
                        image_id=image_id,
                        affordance_index=affordance_index,
                        affordance_item=affordance_item,
                        error_type="parse_error",
                        error_message=parse_error,
                        raw_output=raw_output,
                    )
                    print(f"  Parse error: {parse_error}")
                    print(f"  Raw output: {raw_output}")

                print(f"  Saved to: {args.output_json}")
                print(f"  Error_CSV: {error_csv_path}")
                print(f"  Generation config: {generation_config}")

            except Exception as e:
                append_error_record(
                    error_csv_path=error_csv_path,
                    image_id=image_id,
                    affordance_index=affordance_index,
                    affordance_item=affordance_item,
                    error_type="inference_error",
                    error_message=str(e),
                    raw_output="",
                )
                print(f"[INFERENCE_ERROR] image={image_id} affordance={affordance_index} error={e}")

    print(f"All evaluation finished. Results saved to: {args.output_json}")
    print(f"Error records saved to: {error_csv_path}")


if __name__ == "__main__":
    main()
