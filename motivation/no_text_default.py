import openml
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from tabpfn import TabPFNClassifier


class No_Text:
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
        return accuracy_score(y_test, y_pred)
    
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

        acc = self.test_model(
            X_train.to_numpy(), 
            y_train.to_numpy(), 
            X_test.to_numpy(), 
            y_test.to_numpy()
        )
        
        return acc

def evaluate_default(dataset_ids=[31, 42193, 1461, 1590]):
    encoder = No_Text()
    results = {}
    
    for dataset_id in dataset_ids:
        try:
            acc = encoder.evaluate_dataset(dataset_id)
            if acc is not None:
                results[dataset_id] = acc
                print(f"Dataset {dataset_id}: Accuracy = {acc:.4f}")
        except Exception as e:
            print(f"Error processing dataset {dataset_id}: {str(e)}")
            results[dataset_id] = None
    
    return results

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    print("Evaluating Default No Text:")
    results = evaluate_default()