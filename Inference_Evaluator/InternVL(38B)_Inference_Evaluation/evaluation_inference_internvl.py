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


DEFAULT_CACHE_DIR = "/data03/SUYUKANG/huggingface/hub/"
DEFAULT_MODEL = "OpenGVLab/InternVL3_5-38B-HF"

# Must match the new prompt output order exactly.
# AA-*: Alignment Assessment; PA-*: Plausibility Assessment.
EXPECTED_KEYS = [
    "AA-1", "PA-1",
    "AA-2", "PA-2",
    "AA-3", "PA-3",
    "AA-4", "PA-4",
    "AA-5", "PA-5",
]

INFERENCE_KEYS = [
    "Action purpose",
    "Action duration",
    "Action occurrence type",
    "Action effect",
    "Action effect duration",
]

# Common aliases are kept here so the script can read either the exact prompt
# field names or compact dataset field names without changing the main logic.
INFERENCE_KEY_ALIASES = {
    "Action purpose": ["Action purpose", "action_purpose", "Action_purpose", "Purpose", "purpose"],
    "Action duration": ["Action duration", "action_duration", "Action_duration", "Duration", "duration"],
    "Action occurrence type": [
        "Action occurrence type", "action_occurrence_type", "Action_occurrence_type",
        "Occurrence type", "occurrence_type", "Occurrence", "occurrence",
    ],
    "Action effect": ["Action effect", "action_effect", "Action_effect", "Effect", "effect"],
    "Action effect duration": [
        "Action effect duration", "action_effect_duration", "Action_effect_duration",
        "Effect duration", "effect_duration",
    ],
}


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


def get_first_available(entry: Dict[str, Any], candidate_keys: List[str], default: Any = "") -> Any:
    for key in candidate_keys:
        if key in entry:
            return entry.get(key)
    return default


def normalize_inference_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    core_triplet = entry.get("Core_Triplet", {}) or entry.get("core_triplet", {}) or {}
    if not isinstance(core_triplet, dict):
        core_triplet = {}

    inference_source = entry.get("Affordance-centric inference", None)
    if inference_source is None:
        inference_source = entry.get("Inference", None)
    if inference_source is None:
        inference_source = entry.get("Affordance_Inference", None)
    if inference_source is None:
        inference_source = entry.get("affordance_inference", None)
    if not isinstance(inference_source, dict):
        inference_source = entry

    normalized = {
        "Agent": stringify_compact(entry.get("Agent", core_triplet.get("Agent", ""))),
        "Object": stringify_compact(entry.get("Object", core_triplet.get("Object", ""))),
        "Action": stringify_compact(entry.get("Action", core_triplet.get("Action", ""))),
        "Affordance_Inference": {},
    }

    for key in INFERENCE_KEYS:
        normalized["Affordance_Inference"][key] = stringify_compact(
            get_first_available(inference_source, INFERENCE_KEY_ALIASES[key], "")
        )

    return normalized

def build_prompt(prompt_template: str, inference_entry: Dict[str, Any]) -> str:
    entry = normalize_inference_entry(inference_entry)
    prompt = str(prompt_template)

    replacements = {
        "<Agent>": entry["Agent"],
        "<Object>": entry["Object"],
        "<Objet>": entry["Object"],  # backward compatibility for historical typo
        "<Action>": entry["Action"],
        "<Action purpose>": entry["Affordance_Inference"]["Action purpose"],
        "<Action duration>": entry["Affordance_Inference"]["Action duration"],
        "<Action occurrence type>": entry["Affordance_Inference"]["Action occurrence type"],
        "<Action effect>": entry["Affordance_Inference"]["Action effect"],
        "<Action effect duration>": entry["Affordance_Inference"]["Action effect duration"],
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
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None:
        return ""
    return value


def parse_vlm_json_output(text: str) -> Tuple[Dict[str, Any], Optional[str]]:
    raw_candidate = _extract_json_candidate(text)
    parse_error = None
    parsed: Dict[str, Any] = {}

    try:
        loaded = json.loads(raw_candidate)
        if isinstance(loaded, dict):
            parsed = loaded
        else:
            parse_error = "Model output is not a JSON object."
    except Exception as e:
        parse_error = f"Failed to parse model output as JSON: {e}"

    normalized: Dict[str, Any] = OrderedDict()
    for key in EXPECTED_KEYS:
        normalized[key] = format_evaluator_value(parsed.get(key, ""))

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
            "inference_index",
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
    inference_index: int,
    inference_entry: Optional[Dict[str, Any]],
    error_type: str,
    error_message: str,
    raw_output: str,
) -> None:
    ensure_error_csv_header(error_csv_path)

    agent = str(inference_entry.get("Agent", "")) if inference_entry else ""
    obj = str(inference_entry.get("Object", "")) if inference_entry else ""
    action = str(inference_entry.get("Action", "")) if inference_entry else ""

    with open(error_csv_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            image_id,
            inference_index,
            agent,
            obj,
            action,
            error_type,
            error_message,
            raw_output,
        ])


def init_inference_slot(results: Dict[str, Any], image_id: str, inference_index: int, inference_entry: Dict[str, Any]) -> None:
    normalized_entry = normalize_inference_entry(inference_entry)
    image_bucket = results.setdefault(image_id, {})
    inference_bucket = image_bucket.setdefault(str(inference_index), {})

    inference_bucket["Core_Triplet"] = {
        "Agent": normalized_entry["Agent"],
        "Object": normalized_entry["Object"],
        "Action": normalized_entry["Action"],
    }
    inference_bucket["Affordance_Inference"] = normalized_entry["Affordance_Inference"]

def save_inference_result(
    output_path: str,
    results: Dict[str, Any],
    image_id: str,
    inference_index: int,
    inference_entry: Dict[str, Any],
    evaluator_output: Dict[str, Any],
) -> None:
    init_inference_slot(results, image_id, inference_index, inference_entry)
    bucket = results[image_id][str(inference_index)]
    bucket["Evaluator_Output"] = evaluator_output
    save_results(output_path, results)


def should_skip_existing(results: Dict[str, Any], image_id: str, inference_index: int) -> bool:
    return (
        image_id in results
        and str(inference_index) in results[image_id]
        and isinstance(results[image_id][str(inference_index)], dict)
        and "Evaluator_Output" in results[image_id][str(inference_index)]
    )


def validate_dataset_structure(data: Any) -> List[Dict[str, Any]]:
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of image entries.")
    return data


def get_inference_entries(image_entry: Dict[str, Any]) -> Any:
    """Return candidate inference entries while keeping backward compatibility with Affordance-list input."""
    for key in ["Affordance", "Affordance_Inference", "affordance_inference", "Inference", "inference"]:
        if key in image_entry:
            entries = image_entry.get(key)
            if isinstance(entries, list):
                return entries
            if isinstance(entries, dict):
                core = image_entry.get("Core_Triplet", {}) or {}
                merged = {}
                if isinstance(core, dict):
                    merged.update(core)
                for triplet_key in ["Agent", "Object", "Action"]:
                    if triplet_key in image_entry:
                        merged[triplet_key] = image_entry.get(triplet_key)
                merged["Affordance_Inference"] = entries
                return [merged]
            return entries
    return []


def get_image_id(image_entry: Dict[str, Any]) -> str:
    return stringify_compact(
        image_entry.get("Image_id", image_entry.get("Image_ID", image_entry.get("image_id", "")))
    )

def main():
    parser = argparse.ArgumentParser(description="Evaluate affordance-centric inference with InternVL3.5.")
    parser.add_argument("--images_dir", type=str, default="Data/Framework_Data/Golden Label-image-500", help="Directory containing all images.")
    parser.add_argument("--prompt_py", type=str, default="Data/Framework_Data/prompt.py")
    parser.add_argument("--input_json", type=str, default="Data/Evaluation_Data/Affordance-centric_knowledge_GPT4o_verified_500.json", help="JSON containing Image_id and candidate affordance inference entries.")
    parser.add_argument("--triplet_json", type=str, default=None, help="Backward-compatible alias of --input_json.")
    parser.add_argument("--output_json", type=str, default="Output/output_inference_InternVL3-5(38B)_GPT4o_verified.json")
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
        image_id = get_image_id(image_entry)
        inferences = get_inference_entries(image_entry)

        if not image_id:
            append_error_record(
                error_csv_path=error_csv_path,
                image_id="",
                inference_index=-1,
                inference_entry=None,
                error_type="data_error",
                error_message="Missing Image_id in dataset entry.",
                raw_output="",
            )
            print("[DATA_ERROR] missing Image_id in one dataset entry")
            continue

        if not isinstance(inferences, list):
            append_error_record(
                error_csv_path=error_csv_path,
                image_id=image_id,
                inference_index=-1,
                inference_entry=None,
                error_type="data_error",
                error_message="Affordance field must be a list of candidate inference entries.",
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
                inference_index=-1,
                inference_entry=None,
                error_type="image_error",
                error_message=str(e),
                raw_output="",
            )
            print(f"[IMAGE_ERROR] image={image_id} error={e}")
            continue

        for inference_index, inference_entry in enumerate(inferences):
            init_inference_slot(results, image_id, inference_index, inference_entry)

            if not args.overwrite and should_skip_existing(results, image_id, inference_index):
                print(f"[SKIP] image={image_id} inference={inference_index} already exists in output json")
                continue

            try:
                prompt_text = build_prompt(prompt_template, inference_entry)
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

                save_inference_result(
                    output_path=args.output_json,
                    results=results,
                    image_id=image_id,
                    inference_index=inference_index,
                    inference_entry=inference_entry,
                    evaluator_output=evaluator_output,
                )

                status = "OK" if parse_error is None else "CHECK"
                print(
                    f"[{status}] image={image_id} inference={inference_index} "
                    f"parsed={json.dumps(evaluator_output, ensure_ascii=False)}"
                )

                if parse_error:
                    append_error_record(
                        error_csv_path=error_csv_path,
                        image_id=image_id,
                        inference_index=inference_index,
                        inference_entry=inference_entry,
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
                    inference_index=inference_index,
                    inference_entry=inference_entry,
                    error_type="inference_error",
                    error_message=str(e),
                    raw_output="",
                )
                print(f"[INFERENCE_ERROR] image={image_id} inference={inference_index} error={e}")

    print(f"All evaluation finished. Results saved to: {args.output_json}")
    print(f"Error records saved to: {error_csv_path}")


if __name__ == "__main__":
    main()