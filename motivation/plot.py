import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

# Load evaluation results from JSON
with open("motivation/all_evaluation_results.json", "r") as f:
    results_data = json.load(f)

# Set seaborn style
sns.set_theme(style="whitegrid")

# Consistent colors for each encoding method
colors = {
    "TF-IDF": sns.color_palette("pastel")[0],
    "Word2Vec": sns.color_palette("pastel")[1],
    "FastText": sns.color_palette("pastel")[2],
}

# Baselines
no_text = results_data["No-Text"]["no-text"]["average_accuracy"]
complete_leak = 1.0  # Presumably hardcoded

# Helper function to safely get avg score
def get_avg(group, method_key):
    try:
        return results_data[group][method_key]["average_accuracy"]
    except KeyError:
        return None

# Grouped data extraction from JSON
tfidf_break_scores = [
    ("TF-IDF", get_avg("TF-IDF Breaks", "tf-idf")),
    ("Word2Vec", get_avg("TF-IDF Breaks", "word2vec")),
    ("FastText", get_avg("TF-IDF Breaks", "FT")),
]

w2v_break_scores = [
    ("TF-IDF", get_avg("W2V Breaks", "tf-idf")),
    ("Word2Vec", get_avg("W2V Breaks", "word2vec")),
    ("FastText", get_avg("W2V Breaks", "FT")),
]

fasttext_break_scores = [
    ("TF-IDF", get_avg("Fasttext Breaks", "tf-idf")),
    ("Word2Vec", get_avg("Fasttext Breaks", "word2vec")),
    ("FastText", get_avg("Fasttext Breaks", "FT")),
]

# Create figure and subplots
fig, axs = plt.subplots(1, 4, figsize=(18, 5), gridspec_kw={'width_ratios': [1, 1, 1, 1]})

# Plot baseline (No Text + Complete Leak)
axs[0].bar(["No Text", "Complete Leak"], [no_text, complete_leak], color="lightgray")
axs[0].set_title("Baseline")
axs[0].set_ylim(0, 1.1)
axs[0].set_ylabel("Average Accuracy")
axs[0].set_xticks(["No Text", "Complete Leak"])

# Function to plot each group
def plot_group(ax, data, title):
    for i, (label, value) in enumerate(data):
        if value is not None:
            ax.bar(i, value, color=colors[label], label=label)
    ax.set_title(title)
    ax.set_ylim(0, 1.1)
    ax.set_xticks([])

# Plot the three breakdown groups
plot_group(axs[1], tfidf_break_scores, "N-Grams Break")
plot_group(axs[2], w2v_break_scores, "Simple NLP Break")
plot_group(axs[3], fasttext_break_scores, "LLM Encoding Break")

# Create single legend
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in colors]
labels = list(colors.keys())
fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("motivation/plot.png")
plt.close()