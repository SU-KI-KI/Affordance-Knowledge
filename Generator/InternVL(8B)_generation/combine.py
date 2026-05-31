import json
import os


def merge_json_files(input_files, output_file, deduplicate=True):
    merged_data = []
    seen_image_ids = set()

    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"File not found, skip: {file_path}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"Warning: {file_path} is not a JSON list, skip.")
            continue

        for item in data:
            if not isinstance(item, dict):
                continue

            if deduplicate:
                image_id = item.get("Image_id")
                if image_id in seen_image_ids:
                    continue
                seen_image_ids.add(image_id)

            merged_data.append(item)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"Merged {len(input_files)} files into: {output_file}")
    print(f"Total records: {len(merged_data)}")


if __name__ == "__main__":
    input_files = [
        "Output/out_put_202603200036_1-250.json",
        "Output/out_put_202603200037_251-500.json",
        "Output/out_put_202603202147_1-30.json",
        "Output/out_put_202603211051_1-30.json",
        "Output/out_put_202603221246_1-11.json",
    ]

    output_file = "merged_output.json"

    merge_json_files(input_files, output_file, deduplicate=True)