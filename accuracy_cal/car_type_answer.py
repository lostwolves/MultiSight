import os
import json
import numpy as np
from sklearn.metrics import cohen_kappa_score

def check_label_in_result(json_data):
    """
    Check whether "label" is contained in "result"
    """
    result = json_data.get("result", "").lower()
    label = json_data.get("label", "").lower()
    return label in result

def process_json_files(folder_path, output_file="results.json"):
    """
    Process all JSON files in the folder
    """
    total_files = 0
    correct_files = 0
    results = []
    all_labels = []
    all_preds = []
    unique_labels = set()
    valid_samples = 0

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            total_files += 1
            file_path = os.path.join(folder_path, filename)

            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    json_data = json.load(file)
                    is_correct = check_label_in_result(json_data)
                    if is_correct:
                        correct_files += 1

                    # Record the processing result of the current file
                    results.append({
                        "filename": filename,
                        "is_correct": is_correct,
                        "result": json_data.get("result"),
                        "label": json_data.get("label")
                    })

                    # Collect data for computing the confusion matrix and Kappa coefficient
                    label = json_data.get("label", "").lower()
                    result = json_data.get("result", "").lower()
                    all_labels.append(label)
                    all_preds.append(result)
                    unique_labels.add(label)              

            except Exception as e:
                print(f"Error occurred while processing file {filename}: {e}")

    # Calculate the accuracy
    accuracy = correct_files / total_files if total_files > 0 else 0

    # Manually construct the confusion matrix
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    cm = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)

    for true_label, pred_label in zip(all_labels, all_preds):
        true_index = label_to_index[true_label]
        pred_index = -1
        
        for k,v in label_to_index.items():
            if k in pred_label:
                pred_index = label_to_index[k]
        # if pred_index == -1:
        #     continue
        # else:
        #     valid_samples += 1
        if pred_index == -1:
            if true_index - 1 > 0:
                pred_index = true_index - 1
            else:
                pred_index = true_index + 1
        valid_samples += 1

        cm[true_index, pred_index] += 1

    # Calculate the observed agreement P_o
    total_samples = cm.sum()
    P_o = np.trace(cm) / total_samples

    # Calculate the expected agreement P_e
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    P_e = np.sum(row_sums * col_sums) / (total_samples ** 2)

    # Calculate the Kappa coefficient
    kappa = (P_o - P_e) / (1 - P_e) if (1 - P_e) != 0 else 0

    # Calculate the classification accuracy for each class
    class_accuracies = {}
    for i, label in enumerate(unique_labels):
        class_true = cm[i, i]
        class_total = cm[i, :].sum()
        class_accuracies[label] = class_true / class_total if class_total > 0 else 0

    # Calculate the average classification accuracy
    mean_accuracy = np.mean(list(class_accuracies.values()))

    # Summarize the results
    stats = {
        "total_files": total_files,
        "correct_files": correct_files,
        "accuracy": accuracy,
        "kappa": kappa,
        "class_accuracies": class_accuracies,
        "mean_accuracy": mean_accuracy,
        "valid_samples": valid_samples,
        "confusion_matrix": cm.tolist()
    }

    # Save the results as a JSON file.
    output_data = {
        "statistics": stats,
        "details": results
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"Processing complete! The results have been saved to {output_file}.")

if __name__ == "__main__":

    mid_dir = {"image_train+test_split_cartype_image_grouped" : "results_org.json", 
               "image_train+test_split_cartype_image_grouped_fog" : "results_fog.json",
               "sum" : "results_sum.json"}
    # mid_dir = {"sum" : "results_sum.json"}
    root_dir = "test_answer"
    test_name = "minicpm26"
    read_dir = os.path.join(root_dir, test_name)
    for k,v in mid_dir.items():
        folder_path = os.path.join(read_dir, k)
        output_file = os.path.join(read_dir, v)
        process_json_files(folder_path, output_file)