import json
import function
import prompt
import config
from datetime import datetime

def main():

    start_line = config.start_line
    end_line = config.end_line
    prompt_p = getattr(prompt, config.prompt)

    time = datetime.now().strftime("%Y%m%d%H%M%S")
    URL_file_path = config.image_URL
    json_file_path = config.object_list
    output_file_path = f'Out_put/output_500_{config.start_line}-{config.end_line}_{config.prompt}_{time}.json'

    try:
        function.affordance_extraction(URL_file_path, json_file_path, output_file_path, prompt_p, start_line, end_line)
        print("Affordance extraction completed successfully.")
    except Exception as e:
        print(f"Error occurred during affordance extraction: {e}")


if __name__ == "__main__":
    main()
