import os
import json

def check_label_in_result(json_data):
    """
    Check whether "label" is contained in "result"
    """
    result = json_data.get("result1", "").lower()
    label = json_data.get("label", "").lower()
    return label in result

def process_json_files(folder_path, output_file = "results.json"):
    """
    Process all JSON files in the folder
    """
    total_files = 0
    correct_files = 0
    recall_num = 0
    results = []

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            total_files += 1
            file_path = os.path.join(folder_path, filename)

            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    json_data = json.load(file)
                    recall_num += json_data.get("Recall_n")
                    is_correct = check_label_in_result(json_data)
                    if is_correct:
                        correct_files += 1

                    # Record the processing result of the current file
                    results.append({
                        "filename": filename,
                        "is_correct": is_correct,
                        "result_1": json_data.get("result1"),
                        "label": json_data.get("label"),
                        "Recall_n": json_data.get("Recall_n"),
                    })
            except Exception as e:
                print(f"Error occurred while processing file {filename}: {e}")

    # Summarize the results
    stats = {
        "total_files": total_files,
        "correct_files": correct_files,
        "accuracy": correct_files / total_files if total_files > 0 else 0,
        "recall_n": recall_num / total_files if total_files > 0 else 0
    }

    # Save the results as a JSON file
    output_data = {
        "statistics": stats,
        "details": results
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"Processing completed! Results saved to {output_file}")

if __name__ == "__main__":

    folder_path = input("Please enter the folder path containing JSON files:")
    output_file = "results.json"
    process_json_files(folder_path, output_file)