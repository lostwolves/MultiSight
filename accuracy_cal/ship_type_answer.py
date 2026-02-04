import os
import json
import numpy as np
from sklearn.metrics import cohen_kappa_score

import Levenshtein
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

## Two methods for calculating string similarity
def calculate_similarity_Levenshtein(data):
    """
    Calculate the string similarity between "result" and "label".
    :param data: A dictionary containing the keys "result" and "label".
    :return: A similarity score between 0 and 1.
    """
    result = data.get("result", "")
    label = data.get("label", "")
    if not result or not label:
        return 0.0
    # Calculate the Levenshtein distance
    distance = Levenshtein.distance(result, label)
    # Calculate the similarity (normalized to the range [0, 1])
    max_length = max(len(result), len(label))
    similarity = 1 - (distance / max_length)
    return similarity

def jaccard_similarity(data):
    set1 = set(data["result"])
    set2 = set(data["label"])
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

## Fuse the results of two string similarity metrics to make a final decision
def check_label_in_result(json_data, standard_score):
    score1 = calculate_similarity_Levenshtein(json_data)
    score2 = jaccard_similarity(json_data)
    mean_score = (np.mean([score1, score2]))
    if mean_score > standard_score:
        return True
    else:
        return False

def process_json_files(folder_path, output_file, score, category_list):
    total_files = 0
    correct_files = 0
    results = []
    unique_labels = set()
    acc_category = {}
    acc_min_category = {}

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            total_files += 1
            file_path = os.path.join(folder_path, filename)

            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    json_data = json.load(file)
                    is_correct = check_label_in_result(json_data, score)  # Check whether the classification is correct
                    ## Summarize the classification information for the three major categories
                    if json_data["main_label"] not in acc_category.keys():
                        acc_category[json_data["main_label"]] = {"Sum_num": 0, "Correct_num": 0}
                    acc_category[json_data["main_label"]]["Sum_num"] += 1
                    ## Summarize the classification information for all subcategories
                    if json_data["label"] not in acc_min_category.keys():
                        acc_min_category[json_data["label"]] = {"Sum_num": 0, "Correct_num": 0}
                    acc_min_category[json_data["label"]]["Sum_num"] += 1
                    
                    if is_correct:
                        correct_files += 1
                        acc_category[json_data["main_label"]]["Correct_num"] += 1
                        acc_min_category[json_data["label"]]["Correct_num"] += 1

                    # Record the processing result of the current file
                    results.append({
                        "filename": filename,
                        "is_correct": is_correct,
                        "result": json_data.get("result"),
                        "label": json_data.get("label")
                    })         

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")

    # Calculate the accuracy
    accuracy = correct_files / total_files if total_files > 0 else 0
    
    # Calculate the classification accuracy for each subcategory
    class_accuracies = {}
    for k,v in acc_min_category.items():
        class_true = v["Correct_num"]
        class_total = v["Sum_num"]
        v["Acc"] = class_true / class_total if class_total > 0 else 0
        class_accuracies[k] = v["Acc"]
        acc_min_category[k] = v
        
    # Calculate the classification accuracy for each major category
    for k,v in acc_category.items():
        mid_info = v
        mid_info["Acc"] = v["Correct_num"] / v["Sum_num"]
        acc_category[k] = mid_info
    
    # Calculate the average classification accuracy across all subcategories
    mean_accuracy = np.mean(list(class_accuracies.values()))

    # Summarize the results
    stats = {
        "total_files": total_files,
        "correct_files": correct_files,
        "accuracy": accuracy,
        "class_accuracies": class_accuracies,
        "mean_accuracy": mean_accuracy,
        "category_accuracy" : acc_category,
        "min_category_accuracy" : acc_min_category,
    }

    # Save the results as a JSON file
    output_data = {
        "statistics": stats,
        "details": results
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"Processing complete! The results have been saved to {output_file}")

if __name__ == "__main__":

    folder_path = input("Please enter the folder path containing JSON files: ")
    output_file = "results_MultiSight_prompt.json"
    score = 0.5
    category_list = ["Aircraft_Carrier", "Destroyer", "Frigate"]
    process_json_files(folder_path, output_file, score, category_list)