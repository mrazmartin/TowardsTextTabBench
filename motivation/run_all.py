from motivation.fasttext_breaks import TextEncoder_FT
from motivation.no_text_default import No_Text
from motivation.tf_idf_breaks import TextEncoder_TFIDF
from motivation.w2v_breaks import TextEncoder_WV
import json

def evaluate_encoding_technique(encoder_class, encoding_technique, dataset_ids=[31, 42193, 1461, 1590]):
    encoder = encoder_class(encoding_technique) if encoding_technique else encoder_class()
    results = {}

    for dataset_id in dataset_ids:
        try:
            acc = encoder.evaluate_dataset(dataset_id)
            if acc is not None:
                results[dataset_id] = acc
                print(f"Dataset {dataset_id} with {encoding_technique or 'no_text'}: Accuracy = {acc:.4f}")
        except Exception as e:
            print(f"Error processing dataset {dataset_id}: {str(e)}")
            results[dataset_id] = None

    valid_accuracies = [acc for acc in results.values() if acc is not None]
    avg_accuracy = sum(valid_accuracies) / len(valid_accuracies) if valid_accuracies else None

    print(f"Average Accuracy for {encoding_technique or 'no_text'}: {avg_accuracy:.4f}" if avg_accuracy is not None else "\nNo valid results to compute average.")

    return {
        "average_accuracy": avg_accuracy,
        "per_dataset": results
    }


def collect_all_results():
    all_results = {}

    # Evaluate fasttext
    print("Fasttext Breaks:")
    all_results['Fasttext Breaks'] = {
        "tf-idf" : evaluate_encoding_technique(TextEncoder_FT, 'tf-idf'),
        "word2vec" : evaluate_encoding_technique(TextEncoder_FT, 'word2vec'),
        "FT" : evaluate_encoding_technique(TextEncoder_FT, 'fasttext')
    }

    # Evaluate word2vec
    print("\nW2V Breaks:")
    all_results['W2V Breaks'] = {
        "tf-idf" : evaluate_encoding_technique(TextEncoder_WV, 'tf-idf'),
        "word2vec" : evaluate_encoding_technique(TextEncoder_WV, 'word2vec'),
        "FT" : evaluate_encoding_technique(TextEncoder_WV, 'fasttext')
    }

    # Evaluate tf-idf
    print("\nTF-IDF Breaks")
    all_results['TF-IDF Breaks'] = {
        "tf-idf" : evaluate_encoding_technique(TextEncoder_TFIDF, 'tf-idf'),
        "word2vec" : evaluate_encoding_technique(TextEncoder_TFIDF, 'word2vec'),
        "FT" : evaluate_encoding_technique(TextEncoder_TFIDF, 'fasttext')
    }

    # Evaluate no-text
    print("\nNo Text")
    all_results['No-Text'] = {
        "no-text": evaluate_encoding_technique(No_Text, None)
    }
    return all_results

def save_all_results(all_results, filename="motivation/all_evaluation_results.json"):
    with open(filename, "w") as f:
        json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    import pandas as pd
    pd.set_option('display.max_columns', None)

    results_dict = collect_all_results()
    save_all_results(results_dict)

