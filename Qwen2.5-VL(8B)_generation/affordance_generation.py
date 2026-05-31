from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import prompt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import csv
import json

# ===== User data file paths =====
CSV_PATH = "Data/ALL_URL.csv"      # The CSV column containing image URLs (tries to match: image, url, image_url)
ALL_OBJECTS_PATH = "Data/All_object.json"
OUTPUT_PATH = "Output/output_2025_9_23.json"

MODEL_PATH = "Model/Qwen2.5-VL-7B-Instruct"

llm = LLM(
    model=MODEL_PATH,
    limit_mm_per_prompt={"image": 10, "video": 10},
    tensor_parallel_size=2
)

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    repetition_penalty=1.05,
    max_tokens=2048,
    stop_token_ids=[],
)

# ---- Utility functions ----

def load_object_map(all_objects_path):
    """Read All_object.json and return a dict: image_id -> [objects, ...]"""
    mapping = {}
    with open(all_objects_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        img = item.get("image_id")
        objs = item.get("objects", [])
        if img:
            mapping[img] = objs
    return mapping

def find_image_column(header):
    """Find the image URL column name from the CSV header"""
    candidates = ["image", "url", "image_url", "Image", "URL", "Image_url"]
    for c in candidates:
        if c in header:
            return c
    # Fallback: use the first column
    return header[0] if header else None

def safe_parse_json(s: str):
    """
    Try to parse the model output as JSON.
    If it is wrapped with ```json ... ```, remove the wrapper first.
    """
    if not isinstance(s, str):
        return s

    # Remove Markdown code block markers
    s = s.strip()
    if s.startswith("```json"):
        s = s[len("```json"):].strip()
    if s.startswith("```"):
        s = s[len("```"):].strip()
    if s.endswith("```"):
        s = s[: -len("```")].strip()

    try:
        return json.loads(s)
    except Exception:
        # If parsing fails, return the cleaned raw string
        return s

def read_existing_output(path):
    """Read the existing output array; return an empty array if the file does not exist or is corrupted"""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def write_output_incremental(path, record):
    """
    Write once per iteration:
    read existing array -> append -> write temporary file -> atomically replace,
    preventing data loss if the program crashes midway.
    """
    data = read_existing_output(path)
    data.append(record)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

# ---- Main process ----

def main():
    # 1) Read the image_id -> objects mapping
    id2objs = load_object_map(ALL_OBJECTS_PATH)

    # 2) Prepare the multimodal processor
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # 3) Iterate through each row in the CSV, run inference image by image, and save after each iteration
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        image_col = find_image_column(reader.fieldnames or [])
        if image_col is None:
            raise ValueError("Unable to find the image URL column in the CSV. Expected one of: image, url, image_url")

        for row in reader:
            image_url = row.get(image_col, "").strip()
            if not image_url:
                continue

            image_id = os.path.basename(image_url)
            objects = id2objs.get(image_id, [])
            objects_text = ", ".join(objects) if objects else ""

            # Build the message for this iteration: system comes from prompt.py; user contains the image and corresponding object text
            messages = [
                {"role": "system", "content": prompt.prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_url,
                            "min_pixels": 224 * 224,
                            "max_pixels": 1280 * 28 * 28,
                        },
                        {"type": "text", "text": objects_text},
                    ],
                },
            ]

            # Prepare vLLM input
            chat_prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs

            llm_inputs = {
                "prompt": chat_prompt,
                "multi_modal_data": mm_data,
                "mm_processor_kwargs": video_kwargs,
            }

            # 4) Generate output
            outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
            generated_text = outputs[0].outputs[0].text

            # 5) Try to parse the output as JSON, if the model strictly follows JSON format
            affordance = safe_parse_json(generated_text)

            # 6) Write immediately after each iteration
            rec = {
                "Image_id": image_id,
                "Affordance": affordance,
            }
            write_output_incremental(OUTPUT_PATH, rec)

            # Console message
            print(f"[OK] {image_id} -> appended to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()