import openml
import pandas as pd
import numpy as np
import random
import fasttext
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from tabpfn import TabPFNClassifier
import os
project_root = '/work/dlclarge2/dasb-Camvid/tabadap'

class TextEncoder_BT:
    def __init__(self, encoding_technique):
        self.encoding_technique = encoding_technique
        self.models = self._load_models()
        self.random_words =  [           
            # Positive Synonyms
            "happy", "joyful", "pleased", "delighted", "cheerful", "content", "grateful", "optimistic", "upbeat", "ecstatic",
            "radiant", "thrilled", "hopeful", "enthusiastic", "elated", "blissful", "satisfied", "charming", "agreeable", "nice",
            "awesome", "fabulous", "fantastic", "glorious", "marvelous", "splendid", "superb", "terrific", "admirable", "commendable",
            "excellent", "favorable", "great", "incredible", "lively", "lovely", "magnificent", "outstanding", "peaceful", "refreshing",
            "rejoicing", "serene", "soothing", "supportive", "sympathetic", "tender", "vibrant", "warmhearted", "winsome", "zestful",

            # Negative Synonyms
            "sad", "angry", "upset", "depressed", "bitter", "gloomy", "anxious", "worried", "hostile", "resentful",
            "irritable", "moody", "pessimistic", "fearful", "dismal", "horrible", "terrible", "awful", "nasty", "unpleasant",
            "mean", "cruel", "harsh", "hurtful", "jealous", "malicious", "miserable", "regretful", "repulsive", "scornful",
            "spiteful", "tense", "troubled", "unhappy", "vindictive", "vulgar", "wicked", "wretched", "abrasive", "agonizing",
            "brutal", "callous", "coldhearted", "disrespectful", "evil", "frustrated", "hateful", "hostile", "intolerant", "nervous"
        ]
        
    def _load_models(self):
        models = {}
        if self.encoding_technique == 'fasttext':
            models['fasttext'] = fasttext.load_model(os.path.join(project_root, "embedding_models" ,"fasttext_model", "cc.en.300.bin"))
        elif self.encoding_technique == 'bert':
            models['bert'] = SentenceTransformer("all-MiniLM-L6-v2")
        return models
    
    def get_split(self, X, y):
        X_train = X.sample(frac=0.8, random_state=42) 
        X_test = X.drop(X_train.index)  
        y_train = y.loc[X_train.index]
        y_test = y.loc[X_test.index]
        return X_train, X_test, y_train, y_test
    
    def load_dataset(self, id):
        dataset = openml.datasets.get_dataset(id)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        return X, y
    
    def subsample(self, X, y, num, random_state):
        sampled_indices = X.sample(n=num, random_state=random_state).index
        return X.loc[sampled_indices], y.loc[sampled_indices]
    
    def preprocess_data(self, X, y):
        X = X.dropna()  
        y = y.loc[X.index] 
        y = y.dropna()  
        X = X.loc[y.index] 
        X = self.encode_categorical(X)
        return X, y
    
    def encode_target(self, y, class_A, class_B):
        random.seed(42)

        def generate_sentence(label):
            label = str(label)
            if label == str(class_A):
                keyword = "postive"
            elif label == str(class_B):
                keyword = "negetive"
            else:
                return label  

            num_words = 50
            selected_words = random.sample(self.random_words, num_words)
            insert_index = random.randint(0, num_words)
            selected_words.insert(insert_index, keyword)
            return " ".join(selected_words)

        return y.astype(str).apply(generate_sentence)
    
    def encode_text(self, texts, fit=False):
        if self.encoding_technique == 'tf-idf':
            if fit:
                self.vectorizer = TfidfVectorizer()
                X = self.vectorizer.fit_transform(texts)
                return X.toarray()
            else:
                X = self.vectorizer.transform(texts)
                return X.toarray()
        elif self.encoding_technique == 'fasttext':
            embeddings = [self.models['fasttext'].get_sentence_vector(sent) for sent in texts]
            return np.array(embeddings)
        elif self.encoding_technique == 'bert':
            return self.models['bert'].encode(texts, convert_to_numpy=True)
    
    def reduce_dims(self, vector, dim):
        if dim > vector.shape[1]:
            return vector
        reduced_vector = PCA(n_components=dim).fit_transform(vector)
        return reduced_vector
    
    def encode_categorical(self, X):
        categorical_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
        if categorical_cols:
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X[categorical_cols] = encoder.fit_transform(X[categorical_cols])
        return X
    
    def test_model(self, X_train, y_train, X_test, y_test):
        model = TabPFNClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        return acc
    
    def evaluate_dataset(self, dataset_id, num_samples=100, random_state=42):
        X_whole, y_whole = self.load_dataset(dataset_id)
        X, y = self.subsample(X_whole, y_whole, num=num_samples, random_state=random_state)
        X, y = self.preprocess_data(X, y)
        X_copy, y_copy = X.copy(), y.copy()
        
        binary_targets = y_copy.unique()
        if len(binary_targets) != 2:
            print(f"Dataset {dataset_id} is not binary, skipping...")
            return None
        
        X_train, X_test, y_train, y_test = self.get_split(X_copy, y_copy)   
        target_noisy_train = self.encode_target(y_train, *binary_targets)
        target_noisy_test = self.encode_target(y_test, *binary_targets)

        # Encode the text using the specified technique
        if self.encoding_technique == 'tf-idf':
            target_tf_train = self.encode_text(target_noisy_train.values, fit=True)
            target_tf_test = self.encode_text(target_noisy_test.values, fit=False)
        else:
            target_tf_train = self.encode_text(target_noisy_train.values)
            target_tf_test = self.encode_text(target_noisy_test.values)

        target_tf_df_train = pd.DataFrame(target_tf_train, index=X_train.index)
        target_tf_df_test = pd.DataFrame(target_tf_test, index=X_test.index)

        X_train_noisy = pd.concat([X_train, target_tf_df_train], axis=1)
        X_test_noisy = pd.concat([X_test, target_tf_df_test], axis=1)

        acc = self.test_model(
            X_train_noisy.to_numpy(), 
            y_train.to_numpy(), 
            X_test_noisy.to_numpy(), 
            y_test.to_numpy()
        )
        
        return acc

def evaluate_encoding_technique(encoding_technique, dataset_ids=[31, 42193, 1461, 1590]):
    encoder = TextEncoder_BT(encoding_technique)
    results = {}
    
    for dataset_id in dataset_ids:
        try:
            acc = encoder.evaluate_dataset(dataset_id)
            if acc is not None:
                results[dataset_id] = acc
                print(f"Dataset {dataset_id} with {encoding_technique}: Accuracy = {acc:.4f}")
        except Exception as e:
            print(f"Error processing dataset {dataset_id}: {str(e)}")
            results[dataset_id] = None
    
    return results

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    
    print("Evaluating with tf-idf encoding:")
    tfidf_results = evaluate_encoding_technique('tf-idf')
    
    print("\nEvaluating with fasttext encoding:")
    fasttext_results = evaluate_encoding_technique('fasttext')
    
    print("\nEvaluating with sentence-transformers encoding:")
    st_results = evaluate_encoding_technique('bert')
