from typing import List, Tuple, Dict, Any, Optional

from itertools import product
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split,cross_validate, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.metrics import make_scorer, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import Binarizer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset.

    Parameters:
    - df (pd.DataFrame): The DataFrame to preprocess.

    Returns:
    - pd.DataFrame: The preprocessed DataFrame.
    """
    numerical = df.columns.to_list()
    numerical.remove('strain_id')
    numerical.remove('host_range')

    df[numerical] = df[numerical].clip(lower=0)

    encoded = (df.host_range == 'Broad').astype(int)
    encoded.name = 'host_range_encoded'
    df = pd.concat([df, encoded], axis=1)

    return df,numerical

def calculate_mutual_information(df: pd.DataFrame, features: List[str], target_variable: str = 'host_range_encoded') -> pd.DataFrame:
    """
    Calculates mutual information between each feature and the target variable.

    This function computes the mutual information score for each feature in the given DataFrame 
    relative to a specified target variable. The scores are returned in a sorted DataFrame, 
    with higher scores indicating higher dependency between the feature and the target.

    Parameters
    ----------
    - df (pd.DataFrame): The DataFrame containing the dataset.
    - features (List[str]): A list of column names to be considered for feature selection.
    - target_variable (str, optional): The name of the target variable column. Default is 'host_range_encoded'.

    Returns
    -------
    - pd.DataFrame: A DataFrame with two columns, 'feature' and 'mutual_information', sorted by mutual information in descending order.
    """
    if not set(features).issubset(df.columns):
        raise ValueError("Some features are not present in the DataFrame.")

    X = df[features]
    y = df[target_variable]

    mutual_info = mutual_info_classif(X, y)
    mi_df = pd.DataFrame({'feature': X.columns, 'mutual_information': mutual_info})
    mi_df.sort_values(by='mutual_information', ascending=False, inplace=True)

    return mi_df

def filter_and_select_features(df: pd.DataFrame, numerical_features: List[str], method: str = 'variance', target_variable: str = 'host_range_encoded', threshold: float = 0.01) -> List[str]:
    """
    Filters and selects features from a DataFrame based on a specified method.

    This function applies either 'variance' or 'mutual_information' method to filter and select features
    from the DataFrame. For the 'variance' method, features with variance below a threshold are removed.
    For 'mutual_information', features are selected based on their mutual information score with the target variable.

    Parameters
    ----------
    - df (pd.DataFrame): The DataFrame containing the dataset.
    - numerical_features (List[str]): A list of numerical feature names to be considered for selection.
    - method (str, optional): The method for feature selection ('variance' or 'mutual_information'). Default is 'variance'.
    - target_variable (str, optional): The name of the target variable column. Relevant for 'mutual_information' method. Default is 'host_range_encoded'.
    - threshold (float, optional): The threshold value for feature selection. Default is 0.01.

    Returns
    -------
    - List[str]: A list of selected feature names.
    """
    if not set(numerical_features).issubset(df.columns):
        raise ValueError("Some numerical features are not present in the DataFrame.")

    if method == 'variance':
        features_dict = df[numerical_features].to_dict(orient='records')
        selector = VarianceThreshold(threshold=threshold)

        dv = DictVectorizer(sparse=False)
        features_vectorized = dv.fit_transform(features_dict)
        feature_names = dv.get_feature_names_out()

        features_filtered = selector.fit_transform(features_vectorized)
        support_array = selector.get_support()
        selected_features = feature_names[support_array].tolist()
    
    elif method == 'mutual_information':
        mi_df = calculate_mutual_information(df, numerical_features, target_variable=target_variable)
        selected_features = mi_df[mi_df.mutual_information > threshold].feature.to_list()
    
    else:
        raise ValueError("Invalid method. Choose 'variance' or 'mutual_information'.")

    return selected_features

def get_vectorized_training_data(df: pd.DataFrame, features: list) -> Tuple[DictVectorizer, pd.DataFrame]:
    """
    Vectorizes the specified features of a DataFrame.

    This function takes a DataFrame and a list of feature names, and returns a DictVectorizer object
    and a transformed DataFrame where the specified features are vectorized. This is particularly useful
    for converting categorical features into a format suitable for machine learning models.

    Parameters
    ----------
    - df (pd.DataFrame): The DataFrame containing the dataset.
    - features (list): A list of feature names to be vectorized.

    Returns
    -------
    - Tuple[DictVectorizer, pd.DataFrame]: A tuple containing the DictVectorizer object and the vectorized DataFrame.
    """
    if not set(features).issubset(df.columns):
        raise ValueError("Some features are not present in the DataFrame.")

    data_dicts = df[features].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_vect = dv.fit_transform(data_dicts)

    return dv, X_vect

def hyperparameter_tuning_cv(df_train, y_train, numerical_features, model_type, param_grid, folds=5, metric=roc_auc_score, selection_threshold=0.01, rand_state=11):
    """
    Performs hyperparameter tuning with cross-validation for Decision Tree, Random Forest, or Logistic Regression models.

    This function conducts exhaustive hyperparameter tuning using cross-validation. It supports 
    Decision Tree, Random Forest, and Logistic Regression models. Feature selection based on a 
    variance threshold is applied within each fold of the cross-validation. The function returns 
    the best score, standard deviation of scores, and the best hyperparameters found.

    Parameters
    ----------
    df_train : pd.DataFrame
        The DataFrame containing the training data.
    y_train : np.ndarray
        The target variable for the training data.
    numerical_features : list
        The list of numerical feature names to be used for training.
    model_type : str
        The type of model to be used for training ('decision_tree', 'random_forest', or 'logistic_regression').
    param_grid : dict
        A dictionary defining the grid of hyperparameters to be tested.
    folds : int, optional
        The number of folds for cross-validation (default is 5).
    metric : callable, optional
        The scoring metric function to be used for evaluating model performance (default is roc_auc_score).
    selection_threshold : float, optional
        The threshold for feature selection based on variance (default is 0.01).
    rand_state : int, optional
        The random state for reproducibility (default is 11).

    Returns
    -------
    best_score : float
        The best average score across the cross-validation folds.
    stdev_score : float
        The standard deviation of the scores across the cross-validation folds.
    best_params : dict
        The dictionary of hyperparameters corresponding to the best score.
    """
    kf = KFold(n_splits=folds, shuffle=True, random_state=rand_state)

    best_score = -np.inf
    best_params = None

    # Generate all combinations of parameters
    param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]

    for params in param_combinations:
        fold_scores = []

        for train_index, val_index in kf.split(df_train):
            # Split data
            df_train_fold, df_val_fold = df_train.iloc[train_index][numerical_features], df_train.iloc[val_index][numerical_features]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

            # Select features based on variance threshold
            selected_features = filter_and_select_features(df_train_fold, numerical_features, threshold=selection_threshold)

            dv, X_train_fold = get_vectorized_training_data(df_train_fold, selected_features)

            val_dicts = df_val_fold[selected_features].to_dict(orient='records')
            X_val_fold = dv.transform(val_dicts)

            # Instantiate the model based on model type
            if model_type == 'decision_tree':
                model = DecisionTreeClassifier(**params, random_state=rand_state)
            elif model_type == 'random_forest':
                model = RandomForestClassifier(**params, random_state=rand_state)
            elif model_type == 'logistic_regression':
                model = LogisticRegression(**params, random_state=rand_state, max_iter=1000)
            else:
                raise ValueError("Invalid model type. Choose 'decision_tree', 'random_forest', or 'logistic_regression'.")

            # Train the model
            model.fit(X_train_fold, y_train_fold)

            # Evaluate the model
            y_pred = model.predict_proba(X_val_fold)[:, 1] if model_type == 'logistic_regression' else model.predict(X_val_fold)
            score = metric(y_val_fold, y_pred)
            fold_scores.append(score)

        # Average score across folds for the current set of parameters
        avg_score = np.mean(fold_scores)
        stdev_score = np.std(fold_scores)

        if avg_score > best_score:
            best_score = avg_score
            best_params = params

    return best_score, stdev_score, best_params

def train_final_model_lr(df_train, y_train, features, model_type, best_parameters, rand_state=11):
    """
    Trains a machine learning model using the best parameters found from hyperparameter tuning.

    This function trains a Decision Tree, Random Forest, or Logistic Regression model using the best parameters obtained 
    from a hyperparameter tuning process. It returns the trained model.

    Parameters
    ----------
    df_train : pd.DataFrame
        The DataFrame containing the training data.
    y_train : np.ndarray
        The target variable for the training data.
    features : list
        The list of feature names to be used for training.
    model_type : str
        The type of model to be trained ('decision_tree', 'random_forest', or 'logistic_regression').
    best_parameters : Dict[str, any]
        A dictionary of the best hyperparameters for the model.
    rand_state : int
        The random state for reproducibility.

    Returns
    -------
    Tuple[BaseEstimator, DictVectorizer]
        A tuple containing the trained machine learning model and DictVectorizer object.
    """
    dv, X_train_vect = get_vectorized_training_data(df_train, features)

    if model_type == 'decision_tree':
        model = DecisionTreeClassifier(**best_parameters, random_state=rand_state)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**best_parameters, random_state=rand_state)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(**best_parameters, random_state=rand_state, max_iter=1000)
    else:
        raise ValueError("Invalid model type. Choose 'decision_tree', 'random_forest', or 'logistic_regression'.")

    model.fit(X_train_vect, y_train)
    return model, dv

def test_final_model(model, df_test, y_test, dv, features, threshold=0.5):
    """
    Tests a trained machine learning model on a test dataset and evaluates its performance using various metrics.

    This function vectorizes the test dataset, uses the trained model to make predictions, and calculates 
    the ROC AUC score, precision, recall, and F1 score to evaluate the model's performance.

    Parameters:
    model : BaseEstimator
        The trained machine learning model.
    df_test : pd.DataFrame
        The DataFrame containing the test data.
    y_test : np.ndarray
        The actual target variable values for the test data.
    dv : DictVectorizer
        The DictVectorizer used to vectorize the training data, for consistent transformation of test data.
    features : list
        The list of feature names to be used for testing.
    threshold : float, optional
        The threshold for converting probabilities to binary class predictions (default is 0.5).

    Returns:
    Dict[str, float]
        A dictionary containing the ROC AUC, precision, recall, and F1 score of the model on the test dataset.
    """
    test_dicts = df_test[features].to_dict(orient='records')
    X_test = dv.transform(test_dicts)

    y_pred_proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test,y_pred_proba)
    return roc_auc