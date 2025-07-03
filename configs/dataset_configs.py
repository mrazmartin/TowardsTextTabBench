
import numpy as np
def weighted_loss(y_true, y_pred, class_weights=None):
    if class_weights is None:
        num_classes = y_pred.shape[1]
        class_weights = np.ones(num_classes)
        
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # Convert y_true to int type and flatten if needed
    y_true = y_true.astype(int).to_numpy() if hasattr(y_true, "to_numpy") else np.array(y_true).astype(int)
    
    true_class_probs = y_pred[np.arange(len(y_true)), y_true]
    losses = -np.log(true_class_probs)
    weights = class_weights[y_true]
    weighted_losses = weights * losses
    
    return np.mean(weighted_losses)

data_configs = {
    # classification datasets
    'customer_complaints': {
        'task': 'clf',
        'ntbk': 'complaints_data.ipynb',
        'name': 'customer_complaints'},
#    'diabetes': {crossed by m
#        'task': 'clf',
#        'ntbk': 'diabetes_data.ipynb',
#        'name': 'diabetes'},
    'job_frauds': {
        'task': 'clf',
        'ntbk': 'fraud_detec_data.ipynb',
        'name': 'job_frauds'},
    'hs_cards': {
        'task': 'clf',
        'ntbk': 'HS_cards_data.ipynb',
        'name': 'hs_cards',
        'dataset_name': 'hs_cards'},
    'kickstarter': {
        'task': 'clf',
        'ntbk': 'kickstarter_data.ipynb',
        'name': 'kickstarter'},
#    'lending_club': {crossed by m
#        'task': 'clf',
#        'ntbk': 'lending_club_data.ipynb',
#        'name': 'lending_club'},
#    'okcupid': {crossed by m
#        'task': 'clf',
#        'ntbk': 'okcupid_attr_data.ipynb',
#        'name': 'okcupid'},
    'osha_accidents': {
        'task': 'clf',
        'ntbk': 'OSHA_accidents_data.ipynb',
        'name': 'osha_accidents'},
    'spotify': {
        'task': 'clf',
        'ntbk': 'spotify_genre_data.ipynb',
        'name': 'spotify'},
    # now regression datasets
    'airbnb': {
        'task': 'reg',
        'ntbk': 'airbnb_price_data.ipynb',
        'name': 'airbnb'},
    'beer': {
        'task': 'reg',
        'ntbk': 'beer_rating_data.ipynb',
        'name': 'beer'},
    'calif_houses': {
        'task': 'reg',
        'ntbk': 'calif_houses_data.ipynb',
        'name': 'calif_houses'},
#    'covid_trials': { crossed by m
#        'task': 'reg',
#        'ntbk': 'covid_trials_time_data.ipynb',
#        'name': 'covid_trials'},
#    'drugs_rating': {crossed by m
#        'task': 'reg',
#        'ntbk': 'drugs_rating_data.ipynb',
#        'name': 'drugs_rating'},
#    'insurance_complaints': {crossed by m
#        'task': 'reg',
#        'ntbk': 'ins_complaint_money_data.ipynb',
#        'name': 'insurance_complaints'},
    'it_salary': {
        'task': 'reg',
        'ntbk': 'IT_eu_salary_data_.ipynb',
        'name': 'it_salary'},
    'laptops': {
        'task': 'reg',
        'ntbk': 'laptops_data.ipynb',
        'name': 'laptops'},
    'mercari': {
        'task': 'reg',
        'ntbk': 'mercari_price_data.ipynb',
        'name': 'mercari'},
    'sf_permits': {
        'task': 'reg',
        'ntbk': 'sf_permit_time_data.ipynb',
        'name': 'sf_permits'},
#    'stack_overflow': {
#        'task': 'reg',
#        'ntbk': 'stackoverflow_salary_data.ipynb',
#        'name': 'stack_overflow'},
    'wine': {
        'task': 'reg',
        'ntbk': 'wine_cost_data.ipynb',
        'name': 'wine'},
}



def get_dataset_list(datasets_selection):
    """
    Return a list of dataset names based on the selection criteria.
    
    Args:
        datasets_selection (str): Selection criteria for datasets. Options are:
            - 'all': All datasets
            - 'clf': All classification datasets
            - 'reg': All regression datasets
            - Specific dataset name: e.g., 'customer_complaints'
    """
    if datasets_selection == 'all':
        dataset_name_list = get_all_datasets()
    elif datasets_selection == 'clf':
        dataset_name_list = get_classification_datasets()
    elif datasets_selection == 'reg':
        dataset_name_list = get_regression_datasets()
    elif datasets_selection in get_all_datasets():
        dataset_name_list = [datasets_selection]
    else:
        raise ValueError(f"Invalid selection '{datasets_selection}'. Choose 'all', 'clf', 'reg', or a specific dataset name.")
    return dataset_name_list

def get_a_dataset_dict(name):
    """
    Return a specific dataset configuration by name. Fake it as nested dict.
    """

    return data_configs.get(name, None)

def get_all_datasets():
    """
    Return a list of all dataset names
    """
    return list(data_configs.keys())

def get_regression_datasets():
    """
    Return a list of regression datasets names.
    """

    return [config_key for config_key in data_configs.keys() if data_configs[config_key]['task'] == 'reg']

def get_classification_datasets():
    """
    Return a list of classification datasets names.
    """
    return [config_key for config_key in data_configs.keys() if data_configs[config_key]['task'] == 'clf']


if __name__ == "__main__":
    # Example usage
    print("All datasets:", get_all_datasets())
    print("Regression datasets:", get_regression_datasets())
    print("Classification datasets:", get_classification_datasets())
    print("Specific dataset:", get_a_dataset_dict('customer_complaints'))