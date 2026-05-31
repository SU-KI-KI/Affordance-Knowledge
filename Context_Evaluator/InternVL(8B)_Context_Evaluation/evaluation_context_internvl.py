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
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_CACHE_DIR = "/data/SUYUKANG/huggingface/hub/"
DEFAULT_MODEL = "OpenGVLab/InternVL3_5-8B-HF"

# The current prompt requires these exact JSON output keys and order.
# Keep this list synchronized with the prompt's Output Requirements.
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
    "Persona",
]

BASE_CONTEXT_KEYS = CONTEXT_KEYS[:-1]

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


def normalize_context_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    context = entry.get("Context", {}) or {}
    if not isinstance(context, dict):
        context = {}

    # New prompt format puts Persona inside Context. Keep backward
    # compatibility with older data where Persona was a sibling field.
    raw_persona = context.get("Persona", entry.get("Persona", {}))
    formatted_persona = format_persona_for_prompt(raw_persona)

    normalized_context = {k: stringify_compact(context.get(k, "")) for k in BASE_CONTEXT_KEYS}
    normalized_context["Persona"] = formatted_persona

    normalized = {
        "Agent": str(entry.get("Agent", "")),
        "Object": str(entry.get("Object", "")),
        "Action": str(entry.get("Action", "")),
        "Context": normalized_context,
    }
    return normalized


def build_prompt(prompt_template: str, context_entry: Dict[str, Any]) -> str:
    entry = normalize_context_entry(context_entry)
    prompt = str(prompt_template)

    replacements = {
        "<Agent>": entry["Agent"],
        "<Object>": entry["Object"],
        "<Objet>": entry["Object"],  # backward compatibility for historical typo
        "<Action>": entry["Action"],
        "<Agent-specific context>": entry["Context"]["Agent-specific context"],
        "<Object-specific context>": entry["Context"]["Object-specific context"],
        "<Environmental context>": entry["Context"]["Environmental context"],
        "<Cultural context>": entry["Context"]["Cultural context"],
        "<Temporal context>": entry["Context"]["Temporal context"],
        "<Persona>": entry["Context"]["Persona"],
    }

    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, value)

    return prompt


def load_model_and_processor(
    model_name: str,
    device_map: str,
    dtype: str,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
):
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=dtype_map[dtype],
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=device_map,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    ).eval()

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    return model, processor


def _infer_model_dtype(model) -> torch.dtype:
    dtype = getattr(getattr(model, "config", None), "torch_dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    return torch.bfloat16


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
    ).to(model.device, dtype=_infer_model_dtype(model))

    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
        "temperature": temperature,
        "top_p": top_p,
        "pad_token_id": processor.tokenizer.eos_token_id,
    }

    with torch.inference_mode():
        gen_ids = model.generate(**inputs, **generation_config)

    response = processor.decode(
        gen_ids[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()
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
    """Normalize one evaluator output value before saving.

    Expected shape: [Label, Confidence, Explanation].  Confidence is saved as
    a number.  If the model returns it as a string, convert it to float.
    """
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        normalized = list(value)
        if len(normalized) >= 2 and isinstance(normalized[1], str):
            try:
                normalized[1] = float(normalized[1])
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
            errors.append(f"{key} confidence must be a number, or a numeric string, from 0.00 to 1.00.")
        else:
            if not 0.0 <= confidence_value <= 1.0:
                errors.append(f"{key} confidence must be between 0.00 and 1.00; got {confidence!r}.")

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


def save_results(output_path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
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
            "context_index",
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
    context_index: int,
    context_entry: Optional[Dict[str, Any]],
    error_type: str,
    error_message: str,
    raw_output: str,
) -> None:
    ensure_error_csv_header(error_csv_path)

    agent = str(context_entry.get("Agent", "")) if context_entry else ""
    obj = str(context_entry.get("Object", "")) if context_entry else ""
    action = str(context_entry.get("Action", "")) if context_entry else ""

    with open(error_csv_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            image_id,
            context_index,
            agent,
            obj,
            action,
            error_type,
            error_message,
            raw_output,
        ])


def init_context_slot(results: Dict[str, Any], image_id: str, context_index: int, context_entry: Dict[str, Any]) -> None:
    normalized_entry = normalize_context_entry(context_entry)
    image_bucket = results.setdefault(image_id, {})
    context_bucket = image_bucket.setdefault(str(context_index), {})

    context_bucket["Core_Triplet"] = {
        "Agent": normalized_entry["Agent"],
        "Object": normalized_entry["Object"],
        "Action": normalized_entry["Action"],
    }
    context_bucket["Context"] = normalized_entry["Context"]


def save_context_result(
    output_path: str,
    results: Dict[str, Any],
    image_id: str,
    context_index: int,
    context_entry: Dict[str, Any],
    evaluator_output: Dict[str, Any],
) -> None:
    init_context_slot(results, image_id, context_index, context_entry)
    bucket = results[image_id][str(context_index)]
    bucket["Evaluator_Output"] = evaluator_output
    save_results(output_path, results)


def should_skip_existing(results: Dict[str, Any], image_id: str, context_index: int) -> bool:
    return (
        image_id in results
        and str(context_index) in results[image_id]
        and isinstance(results[image_id][str(context_index)], dict)
        and "Evaluator_Output" in results[image_id][str(context_index)]
    )


def validate_dataset_structure(data: Any) -> List[Dict[str, Any]]:
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of image entries.")
    return data


def main():
    parser = argparse.ArgumentParser(description="Evaluate affordance context RA/IA outputs with InternVL3.5.")
    parser.add_argument("--images_dir", type=str, default="Data/Framework_Data/Golden Label-image-500", help="Directory containing all images.")
    parser.add_argument("--prompt_py", type=str, default="Data/Framework_Data/prompt.py")
    parser.add_argument("--input_json", type=str, default="Data/Evaluation_Data/Affordance-centric_knowledge_GPT4o_verified_500.json", help="JSON containing Image_id and Affordance entries.")
    parser.add_argument("--triplet_json", type=str, default=None, help="Backward-compatible alias of --input_json.")
    parser.add_argument("--output_json", type=str, default="Output/output_context_InternVL3-5(8B)_ChatGPT4o_verified.json")
    parser.add_argument("--error_csv", type=str, default=None)
    parser.add_argument("--model", "--model_name", dest="model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit_images", type=int, default=None)
    args = parser.parse_args()

    if args.triplet_json and not args.input_json:
        args.input_json = args.triplet_json
    elif args.triplet_json:
        args.input_json = args.triplet_json

    error_csv_path = args.error_csv or get_default_error_csv_path(args.output_json)

    if args.overwrite and os.path.exists(args.output_json):
        os.remove(args.output_json)
    if args.overwrite and os.path.exists(error_csv_path):
        os.remove(error_csv_path)

    prompt_template = load_prompt_template(args.prompt_py)
    dataset = validate_dataset_structure(load_json(args.input_json))

    if args.limit_images is not None:
        dataset = dataset[: args.limit_images]

    model, processor = load_model_and_processor(
        model_name=args.model,
        device_map=args.device_map,
        dtype=args.dtype,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )

    results = {} if args.overwrite else read_existing_results(args.output_json)

    for image_entry in dataset:
        image_id = image_entry.get("Image_id", "")
        contexts = image_entry.get("Affordance", [])

        if not image_id:
            append_error_record(
                error_csv_path=error_csv_path,
                image_id="",
                context_index=-1,
                context_entry=None,
                error_type="data_error",
                error_message="Missing Image_id in dataset entry.",
                raw_output="",
            )
            print("[DATA_ERROR] missing Image_id in one dataset entry")
            continue

        if not isinstance(contexts, list):
            append_error_record(
                error_csv_path=error_csv_path,
                image_id=image_id,
                context_index=-1,
                context_entry=None,
                error_type="data_error",
                error_message="Affordance field must be a list.",
                raw_output="",
            )
            print(f"[DATA_ERROR] image={image_id} Affordance is not a list")
            continue

        try:
            image_path = resolve_image_path(args.images_dir, image_id)
        except Exception as e:
            append_error_record(
                error_csv_path=error_csv_path,
                image_id=image_id,
                context_index=-1,
                context_entry=None,
                error_type="image_error",
                error_message=str(e),
                raw_output="",
            )
            print(f"[IMAGE_ERROR] image={image_id} error={e}")
            continue

        for context_index, context_entry in enumerate(contexts):
            init_context_slot(results, image_id, context_index, context_entry)

            if not args.overwrite and should_skip_existing(results, image_id, context_index):
                print(f"[SKIP] image={image_id} context={context_index} already exists in output json")
                continue

            try:
                prompt_text = build_prompt(prompt_template, context_entry)
                # print (prompt_text)
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

                save_context_result(
                    output_path=args.output_json,
                    results=results,
                    image_id=image_id,
                    context_index=context_index,
                    context_entry=context_entry,
                    evaluator_output=evaluator_output,
                )

                status = "OK" if parse_error is None else "CHECK"
                print(
                    f"[{status}] image={image_id} context={context_index} "
                    f"parsed={json.dumps(evaluator_output, ensure_ascii=False)}"
                )

                if parse_error:
                    append_error_record(
                        error_csv_path=error_csv_path,
                        image_id=image_id,
                        context_index=context_index,
                        context_entry=context_entry,
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
                    context_index=context_index,
                    context_entry=context_entry,
                    error_type="inference_error",
                    error_message=str(e),
                    raw_output="",
                )
                print(f"[INFERENCE_ERROR] image={image_id} context={context_index} error={e}")

    print(f"All evaluation finished. Results saved to: {args.output_json}")
    print(f"Error records saved to: {error_csv_path}")


if __name__ == "__main__":
    main()