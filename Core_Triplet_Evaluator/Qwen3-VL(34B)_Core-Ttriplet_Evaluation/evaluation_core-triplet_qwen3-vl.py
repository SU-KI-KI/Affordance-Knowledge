import argparse
import csv
import importlib.util
import json
import os
import re
from typing import Any, Dict, Optional, Tuple

from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


DEFAULT_CACHE_DIR = "/data03/SUYUKANG/huggingface/hub/"
DEFAULT_MODEL = os.environ.get("QWEN3_VL_MODEL", "Qwen/Qwen3-VL-32B-Instruct")
EXPECTED_KEYS = ["Q1", "Q2", "Q3", "Q4", "Q5"]


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


def build_prompt(prompt_template: str, triplet: Dict[str, Any]) -> str:
    prompt = str(prompt_template)
    replacements = {
        "<Agent>": str(triplet["Agent"]),
        "<Object>": str(triplet["Object"]),
        "<Objet>": str(triplet["Object"]),  # 兼容历史 typo；确认不需要可删
        "<Action>": str(triplet["Action"]),
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
    """
    保留真正的 JSON 结构：
    - list -> list
    - tuple -> list
    - None -> ""
    - 其他保持原样
    """
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None:
        return ""
    return value


def _q3_first_element_is_irrelevant(q3_value: Any) -> bool:
    target = "Irrelevant to this image"

    if isinstance(q3_value, str):
        return q3_value == target

    if isinstance(q3_value, (list, tuple)) and len(q3_value) > 0:
        first = q3_value[0]
        return isinstance(first, str) and first == target

    return False


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

    normalized: Dict[str, Any] = {}
    for key in EXPECTED_KEYS:
        normalized[key] = format_evaluator_value(parsed.get(key, ""))

    # 规则：
    # 如果 Q3 是字符串且等于 "Irrelevant to this image"
    # 或者 Q3 的第一个元素等于 "Irrelevant to this image"
    # 则 Q4-Q5 全部置空
    if _q3_first_element_is_irrelevant(normalized.get("Q3", "")):
        for key in ["Q4", "Q5"]:
            normalized[key] = ""

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
            "triplet_index",
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
    triplet_index: int,
    triplet: Optional[Dict[str, Any]],
    error_type: str,
    error_message: str,
    raw_output: str,
) -> None:
    ensure_error_csv_header(error_csv_path)

    agent = str(triplet.get("Agent", "")) if triplet else ""
    obj = str(triplet.get("Object", "")) if triplet else ""
    action = str(triplet.get("Action", "")) if triplet else ""

    with open(error_csv_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            image_id,
            triplet_index,
            agent,
            obj,
            action,
            error_type,
            error_message,
            raw_output,
        ])


def init_triplet_slot(results: Dict[str, Any], image_id: str, triplet_index: int, triplet: Dict[str, Any]) -> None:
    image_bucket = results.setdefault(image_id, {})
    triplet_bucket = image_bucket.setdefault(str(triplet_index), {})
    triplet_bucket["Triplet"] = {
        "Agent": str(triplet["Agent"]),
        "Object": str(triplet["Object"]),
        "Action": str(triplet["Action"]),
    }


def save_triplet_result(
    output_path: str,
    results: Dict[str, Any],
    image_id: str,
    triplet_index: int,
    triplet: Dict[str, Any],
    evaluator_output: Dict[str, Any],
) -> None:
    init_triplet_slot(results, image_id, triplet_index, triplet)
    bucket = results[image_id][str(triplet_index)]
    bucket["Evaluator_Output"] = evaluator_output
    save_results(output_path, results)


def main():
    parser = argparse.ArgumentParser(description="Evaluate action triplets with Qwen3-VL.")
    parser.add_argument("--images_dir", type=str, default="Data/Framework_Data/Golden Label-image-500", help="Directory containing all images.")
    parser.add_argument("--prompt_py", type=str, default="Data/Framework_Data/prompt_core-triplet.py")
    parser.add_argument("--triplet_json", type=str, default="Data/Evaluation_Data/Affordance-centric_knowledge_InternVL3-5(38B)_500.json")
    parser.add_argument("--output_json", type=str, default="Output/output_Qwen3-VL(34B)_InternVL3-5(38B).json")
    parser.add_argument("--error_csv", type=str, default=None)
    parser.add_argument("--model", "--model_name", dest="model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--use_flash_attention_2", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=512)
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
    triplet_data = load_json(args.triplet_json)

    if args.limit_images is not None:
        triplet_data = triplet_data[: args.limit_images]

    model, processor = load_model_and_processor(
        model_name=args.model,
        device_map=args.device_map,
        dtype=args.dtype,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        use_flash_attention_2=args.use_flash_attention_2,
    )

    results = {} if args.overwrite else read_existing_results(args.output_json)

    for image_entry in triplet_data:
        image_id = image_entry["Image_id"]
        affordances = image_entry.get("Affordance", [])

        try:
            image_path = resolve_image_path(args.images_dir, image_id)
        except Exception as e:
            append_error_record(
                error_csv_path=error_csv_path,
                image_id=image_id,
                triplet_index=-1,
                triplet=None,
                error_type="image_error",
                error_message=str(e),
                raw_output="",
            )
            print(f"[IMAGE_ERROR] image={image_id} error={e}")
            continue

        for triplet_index, triplet in enumerate(affordances):
            init_triplet_slot(results, image_id, triplet_index, triplet)

            try:
                prompt_text = build_prompt(prompt_template, triplet)
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

                save_triplet_result(
                    output_path=args.output_json,
                    results=results,
                    image_id=image_id,
                    triplet_index=triplet_index,
                    triplet=triplet,
                    evaluator_output=evaluator_output,
                )

                status = "OK" if parse_error is None else "CHECK"
                print(
                    f"[{status}] image={image_id} triplet={triplet_index} "
                    f"parsed={json.dumps(evaluator_output, ensure_ascii=False)}"
                )

                if parse_error:
                    append_error_record(
                        error_csv_path=error_csv_path,
                        image_id=image_id,
                        triplet_index=triplet_index,
                        triplet=triplet,
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
                    triplet_index=triplet_index,
                    triplet=triplet,
                    error_type="inference_error",
                    error_message=str(e),
                    raw_output="",
                )
                print(f"[INFERENCE_ERROR] image={image_id} triplet={triplet_index} error={e}")

    print(f"All evaluation finished. Results saved to: {args.output_json}")
    print(f"Error records saved to: {error_csv_path}")


if __name__ == "__main__":
    main()
