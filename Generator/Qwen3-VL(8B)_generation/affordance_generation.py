import os
import csv
import json
import ast
import re
import importlib.util
from datetime import datetime
from typing import Any, Dict, List, Tuple

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

import config


# -------------------------
# File loading helpers
# -------------------------
def load_prompt_from_py(prompt_py_path: str) -> str:
    spec = importlib.util.spec_from_file_location("user_prompt_module", prompt_py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load prompt file: {prompt_py_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "prompt"):
        raise AttributeError(f"No variable named `prompt` found in {prompt_py_path}")

    prompt = getattr(module, "prompt")
    if not isinstance(prompt, str):
        raise TypeError(f"`prompt` in {prompt_py_path} must be a string.")

    return prompt.strip()


def load_all_urls(csv_path: str) -> List[str]:
    urls: List[str] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            url = row[0].strip()
            if url:
                urls.append(url)
    return urls


def load_object_mapping(json_path: str) -> Dict[str, List[str]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError("All_object.json must be a list of dicts.")

    mapping: Dict[str, List[str]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        image_id = item.get("image_id")
        objects = item.get("objects", [])
        if isinstance(image_id, str):
            mapping[image_id] = objects if isinstance(objects, list) else []

    return mapping


# -------------------------
# Output / log helpers
# -------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M")


def save_json_atomic(data: Any, save_path: str) -> None:
    tmp_path = save_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, save_path)


def append_jsonl(record: Dict[str, Any], path: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_file_paths(timestamp: str, start_row: int, end_row: int) -> Tuple[str, str, str, str]:
    row_range = f"{start_row}-{end_row}"

    output_dir = "Output"
    log_dir = "Log"
    ensure_dir(output_dir)
    ensure_dir(log_dir)

    output_path = os.path.join(output_dir, f"out_put_{timestamp}_{row_range}.json")
    failure_url_path = os.path.join(log_dir, f"log_failure_url_{timestamp}_{row_range}.jsonl")
    failure_info_path = os.path.join(log_dir, f"log_failure_info_{timestamp}_{row_range}.jsonl")
    summary_path = os.path.join(log_dir, f"log_summary_{timestamp}_{row_range}.json")

    return output_path, failure_url_path, failure_info_path, summary_path


# -------------------------
# Prompt / parsing helpers
# -------------------------
def build_user_text(base_prompt: str, objects: List[str]) -> str:
    object_text = json.dumps(objects, ensure_ascii=False)
    return f"{base_prompt}\n\n{object_text}"


def extract_json_text(raw_text: str) -> str:
    text = raw_text.strip()

    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
        return text

    start_obj = text.find("{")
    start_arr = text.find("[")
    candidates = [i for i in [start_obj, start_arr] if i != -1]
    if not candidates:
        raise ValueError("No JSON object/array found in model output.")
    start = min(candidates)

    opening = text[start]
    closing = "}" if opening == "{" else "]"

    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == opening:
                depth += 1
            elif ch == closing:
                depth -= 1
                if depth == 0:
                    return text[start: idx + 1]

    raise ValueError("Failed to extract a complete JSON object/array from model output.")


def parse_model_output(raw_text: str) -> List[Dict[str, Any]]:
    json_text = extract_json_text(raw_text)

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        data = ast.literal_eval(json_text)

    # 情况1：已经是 list，直接返回
    if isinstance(data, list):
        return data

    # 情况2：顶层是 dict
    if isinstance(data, dict):
        # 如果模型包了一层 "Affordance"
        if "Affordance" in data:
            affordance_data = data["Affordance"]
            if isinstance(affordance_data, list):
                return affordance_data
            if isinstance(affordance_data, dict):
                return [affordance_data]
            raise TypeError("`Affordance` field is neither a list nor a dict.")

        # 如果就是单条 affordance dict，强制包成 list
        return [data]

    raise TypeError("Parsed model output is neither a list nor a dict.")


# -------------------------
# Model helpers
# -------------------------
def load_model_and_processor(
    model_name: str,
    cache_dir: str,
    use_flash_attention_2: bool = False,
):
    model_kwargs = {
        "cache_dir": cache_dir,
        "local_files_only": False,
        "device_map": "auto",
    }

    if use_flash_attention_2:
        try:
            import flash_attn  # noqa: F401
            model_kwargs["dtype"] = torch.bfloat16
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
    )
    processor = AutoProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=False,
    )
    return model, processor


def generate_one(
    model,
    processor,
    image_url: str,
    user_text: str,
    max_new_tokens: int = 4096,
    do_sample: bool = False,
    temperature: float = 0.2,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_url},
                {"type": "text", "text": user_text},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "use_cache": True,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generate_kwargs)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


# -------------------------
# Main pipeline
# -------------------------
def main() -> None:
    start_row = config.START_ROW
    end_row = config.END_ROW

    if start_row < 1 or end_row < start_row:
        raise ValueError("START_ROW and END_ROW are invalid. They must satisfy: START_ROW >= 1 and END_ROW >= START_ROW.")

    timestamp = now_timestamp()
    output_path, failure_url_path, failure_info_path, summary_path = build_file_paths(
        timestamp=timestamp,
        start_row=start_row,
        end_row=end_row,
    )

    print("Loading prompt...")
    base_prompt = load_prompt_from_py(config.PROMPT_PY)

    print("Loading url list...")
    all_urls = load_all_urls(config.URL_CSV)
    if not all_urls:
        raise RuntimeError(f"No URLs found in {config.URL_CSV}")

    print("Loading object mapping...")
    object_mapping = load_object_mapping(config.OBJECT_JSON)

    start_index = start_row - 1
    end_index_exclusive = end_row
    urls_to_process = all_urls[start_index:end_index_exclusive]

    if not urls_to_process:
        raise RuntimeError("No URLs selected for processing. Please check START_ROW and END_ROW in config.py")

    print("Loading model and processor...")
    model, processor = load_model_and_processor(
        model_name=config.MODEL_NAME,
        cache_dir=config.CACHE_DIR,
        use_flash_attention_2=getattr(config, "USE_FLASH_ATTENTION_2", False),
    )

    results: List[Dict[str, Any]] = []

    success_count = 0
    failure_count = 0
    total_count = len(urls_to_process)

    for absolute_index, image_url in enumerate(urls_to_process, start=start_row):
        image_id = image_url.rstrip("/").split("/")[-1]
        objects = object_mapping.get(image_id, [])

        print(f"[{absolute_index}] Processing {image_id} ...")
        user_text = build_user_text(base_prompt, objects)

        raw_output = ""
        try:
            raw_output = generate_one(
                model=model,
                processor=processor,
                image_url=image_url,
                user_text=user_text,
                max_new_tokens=getattr(config, "MAX_NEW_TOKENS", 4096),
                do_sample=getattr(config, "DO_SAMPLE", False),
                temperature=getattr(config, "TEMPERATURE", 0.2),
            )

            affordance = parse_model_output(raw_output)
            record = {
                "Image_id": image_id,
                "Affordance": affordance,
            }
            results.append(record)
            success_count += 1

            save_json_atomic(results, output_path)
            print(f"[{absolute_index}] Success on {image_id}. Saved {len(results)} results to: {output_path}")

        except Exception as e:
            failure_count += 1
            print(f"[{absolute_index}] Error on {image_id}: {e}")

            append_jsonl(
                {
                    "row": absolute_index,
                    "image_id": image_id,
                    "image_url": image_url,
                },
                failure_url_path,
            )

            append_jsonl(
                {
                    "row": absolute_index,
                    "image_id": image_id,
                    "image_url": image_url,
                    "objects": objects,
                    "raw_output": raw_output,
                    "error": str(e),
                },
                failure_info_path,
            )

    summary = {
        "start_row": start_row,
        "end_row": end_row,
        "total_selected": total_count,
        "success_count": success_count,
        "failure_count": failure_count,
        "output_path": output_path,
        "failure_url_log_path": failure_url_path,
        "failure_info_log_path": failure_info_path,
    }
    save_json_atomic(summary, summary_path)

    print("=" * 80)
    print(f"Done. Row range: {start_row}-{end_row}")
    print(f"Successful results: {success_count}")
    print(f"Failed results: {failure_count}")
    print(f"Output saved to: {output_path}")
    print(f"Failure url log saved to: {failure_url_path}")
    print(f"Failure info log saved to: {failure_info_path}")
    print(f"Summary log saved to: {summary_path}")


if __name__ == "__main__":
    main()