PARAMETERS = {
    'data_path':'../preprocessed_data/pre-processed_strain_profiles.csv',
    'model_type': 'decision_tree',
    'feature_selection_method':'variance',
    'threshold': 0.01,
    'metric':'roc_auc',
    'model_version':'v1',
    'save_model':True,
    'output_path':'../trained_models'
}

hyperparam_grid = {
        'logistic_regression':{'C':[1,0.9,0.8,0.7,0.6,0.4,0.2]},
        'decision_tree': {'max_depth': [None, 2, 4, 6, 8], 'min_samples_leaf': [1, 2, 3, 4]},
        'random_forest': {'n_estimators': [10, 50, 100], 'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]}
    }