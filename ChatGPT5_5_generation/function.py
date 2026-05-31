import openai
import json
import function
import prompt
import os
import config
import config_single
from datetime import datetime

# Set API key
client = openai.Client(
    api_key="")


# Function for interacting with GPT
def generate_image_affordance(prompt, object_names, image_path):
    try:
        # Call GPT's multimodal interface
        response = client.chat.completions.create(
            model="gpt-5.5",
            messages=[
                {"role": "system", "content": [{"type": "text", "text": prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": object_names},
                        {"type": "image_url", "image_url": {"url": f"{image_path}"}}
                    ]
                }
            ],
            reasoning_effort="low",
            verbosity="low",
            # max_completion_tokens=3072
        )
        print(response)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"


def affordance_extraction(URL_file_path, json_file_path, output_file_path, prompt, start_line=1, end_line=None):
    try:
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
            image_object_map = {item['image_id']: ", ".join(item['objects']) for item in data}
    except FileNotFoundError:
        print(f"Error: File {json_file_path} not found.")
        image_object_map = {}
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {json_file_path}.")
        image_object_map = {}

    # Initialize failure counter
    failure_count = 0
    max_failures = 10
    total_failure_count = 0
    # Count successfully processed entries
    success_count = 0

    dataset_name_URL = os.path.splitext(os.path.basename(URL_file_path))[0]
    dataset_name = dataset_name = dataset_name_URL.split('_')[0]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Create log folder
    log_folder = f"Log/Log_{dataset_name}_{config.start_line}-{config.end_line}_{config.prompt}_{timestamp}"
    os.makedirs(log_folder, exist_ok=True)
    # Create log files
    log_URL = os.path.join(log_folder, f"{dataset_name}_URL_{timestamp}.txt")
    log_URL_failure = os.path.join(log_folder, f"{dataset_name}_URL_failure_{timestamp}.txt")
    log_summary = os.path.join(log_folder, f"{dataset_name}_summary_{timestamp}.txt")

    try:
        with (open(URL_file_path, 'r') as txt_file,
              open(output_file_path, 'w') as output_file,
              open(log_URL, 'w') as log_file,
              open(log_URL_failure, 'w') as log_file_failure):

            lines = txt_file.readlines()

            # Check whether end_line exceeds the total number of lines
            if end_line is None or end_line > len(lines):
                end_line = len(lines)

            # Only process lines within the specified range
            lines_to_process = lines[start_line - 1:end_line]

            output_file.write("[")  # Start JSON array
            first_entry = True

            for line in lines_to_process:
                if failure_count >= max_failures:
                    print(f"Error: Extraction failed {max_failures} times consecutively. Pausing execution.")
                    break  # Stop processing

                url = line.strip()
                image_id = str(url.split('/')[-1])
                objects = image_object_map.get(image_id, None)

                raw_affordance = function.generate_image_affordance(prompt, objects, url)

                try:
                    # Extract affordance information in JSON format from ChatGPT's output
                    if raw_affordance.startswith("```json") and raw_affordance.endswith("```"):
                        raw_affordance = raw_affordance[7:-3].strip()

                    json_affordance = json.loads(raw_affordance)
                    failure_count = 0  # Reset failure counter
                    success_count += 1  # Increase successful processing count
                    log_file.write(f"{url}\n")  # Record the successful URL in the log file

                except json.JSONDecodeError:
                    json_affordance = "0"  # If extraction fails, set it to "0"
                    failure_count += 1  # Increase failure counter
                    total_failure_count += 1  # Increase total failure counter
                    log_file_failure.write(f"{url}\n")

                result_entry = {
                    "Image_id": image_id,
                    "Object": objects,
                    "Affordance": json_affordance
                }

                # Write the result to the file
                if not first_entry:
                    output_file.write(",\n")
                output_file.write(json.dumps(result_entry, indent=4))
                first_entry = False

                print(f"Processed Image: {image_id}, Object: {objects}, Affordance: {json_affordance}")

            output_file.write("\n]")  # End JSON array

    except Exception as e:
        print(f"Error: {e}")

    finally:
        try:
            with open(log_summary, 'w') as summary_log_file:
                summary_log_file.write(f"Dataset: {json_file_path}\n")
                summary_log_file.write(f"Start_line: {config.start_line}\n")
                summary_log_file.write(f"End_line: {config.end_line}\n")
                summary_log_file.write(f"Total URLs processed: {success_count}\n")
                summary_log_file.write(f"Total Failed URLs: {total_failure_count}\n")
                summary_log_file.write(f"Prompt: {config.prompt}\n")
                summary_log_file.write(f"Output file: {output_file_path}\n")
            print(f"Summary log saved to {log_summary}")
        except Exception as e:
            print(f"Error writing summary log: {e}")


def affordance_extraction_single(URL_file_path, json_file_path, output_file_path, prompt, lines_to_extract=None):
    try:
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
            image_object_map = {item['image_id']: ", ".join(item['objects']) for item in data}
    except FileNotFoundError:
        print(f"Error: File {json_file_path} not found.")
        image_object_map = {}
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {json_file_path}.")
        image_object_map = {}

    # Initialize failure counter
    failure_count = 0
    max_failures = 10
    total_failure_count = 0
    success_count = 0

    dataset_name_URL = os.path.splitext(os.path.basename(URL_file_path))[0]
    dataset_name = dataset_name_URL.split('_')[0]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Create log folder
    log_folder = f"Log/Log_{dataset_name}_lines_to_extract_{timestamp}"
    os.makedirs(log_folder, exist_ok=True)

    log_URL = os.path.join(log_folder, f"{dataset_name}_URL_{timestamp}.txt")
    log_URL_failure = os.path.join(log_folder, f"{dataset_name}_URL_failure_{timestamp}.txt")
    log_summary = os.path.join(log_folder, f"{dataset_name}_summary_{timestamp}.txt")

    try:
        with (open(URL_file_path, 'r') as txt_file,
              open(output_file_path, 'w') as output_file,
              open(log_URL, 'w') as log_file,
              open(log_URL_failure, 'w') as log_file_failure):

            lines = txt_file.readlines()

            if lines_to_extract is None:
                print("Error: No lines specified for extraction.")
                return

            # Extract the specified lines
            lines_to_process = [lines[line - 1] for line in lines_to_extract if 1 <= line <= len(lines)]

            output_file.write("[")  # Start JSON array
            first_entry = True

            for line in lines_to_process:
                if failure_count >= max_failures:
                    print(f"Error: Extraction failed {max_failures} times consecutively. Pausing execution.")
                    break  # Stop processing

                url = line.strip()
                image_id = str(url.split('/')[-1])
                objects = image_object_map.get(image_id, None)

                raw_affordance = function.generate_image_affordance(prompt, objects, url)

                try:
                    if raw_affordance.startswith("```json") and raw_affordance.endswith("```"):
                        raw_affordance = raw_affordance[7:-3].strip()

                    json_affordance = json.loads(raw_affordance)
                    failure_count = 0  # Reset failure counter
                    success_count += 1  # Increase successful processing count
                    log_file.write(f"{url}\n")  # Record the successful URL in the log file

                except json.JSONDecodeError:
                    json_affordance = "0"  # If extraction fails, set it to "0"
                    failure_count += 1  # Increase failure counter
                    total_failure_count += 1  # Increase total failure counter
                    log_file_failure.write(f"{url}\n")

                result_entry = {
                    "Image_id": image_id,
                    "Object": objects,
                    "Affordance": json_affordance
                }

                if not first_entry:
                    output_file.write(",\n")
                output_file.write(json.dumps(result_entry, indent=4))
                first_entry = False

                print(f"Processed Image: {image_id}, Object: {objects}, Affordance: {json_affordance}")

            output_file.write("\n]")  # End JSON array

    except Exception as e:
        print(f"Error: {e}")

    finally:
        try:
            with open(log_summary, 'w') as summary_log_file:
                summary_log_file.write(f"Dataset: {json_file_path}\n")
                summary_log_file.write(f"Lines to extract: {lines_to_extract}\n")
                summary_log_file.write(f"Total URLs processed: {success_count}\n")
                summary_log_file.write(f"Total Failed URLs: {total_failure_count}\n")
                summary_log_file.write(f"Prompt: {config_single.prompt}\n")
                summary_log_file.write(f"Output file: {output_file_path}\n")
            print(f"Summary log saved to {log_summary}")
        except Exception as e:
            print(f"Error writing summary log: {e}")