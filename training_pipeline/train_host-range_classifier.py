from ml_utils import *
import logging
from config import PARAMETERS, hyperparam_grid

import pickle

from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Unpack parameters
    data_path = PARAMETERS['data_path']
    model_type = PARAMETERS['model_type']
    feature_selection_method = PARAMETERS['feature_selection_method']
    threshold=PARAMETERS['threshold']
    save_model = PARAMETERS['save_model']
    model_version=PARAMETERS['model_version']
    param_grid = hyperparam_grid[model_type]
    output_path = PARAMETERS['output_path']

    logging.info('Loading data')
    # Load and preprocess data
    df = pd.read_csv(data_path)
    strain_data,numerical = preprocess_data(df)
    

    # Split data
    df_train, df_test = train_test_split(strain_data, test_size=0.2, random_state=11)
    y_train = df_train['host_range_encoded'].values
    y_test = df_test['host_range_encoded'].values

    # Select features for training
    selected_features = filter_and_select_features(df_train, numerical, method=feature_selection_method,threshold=threshold)

    # Hyperparameter tuning and cross validation
    logging.info(f'Performing hyperparameter tuning and cross-validation on {model_type}')
    best_score,std_score,best_params = hyperparameter_tuning_cv(df_train,y_train,numerical,model_type,param_grid=param_grid,metric=roc_auc_score)
    logging.info(f'Best mean score: {best_score}, stdev: {std_score}')
    logging.info(f'Best hyperparameters: {best_params}')

    # Train final model based on best hyperparameters   
    logging.info('Training and testing final model') 
    final_model,dv = train_final_model(df_train, y_train, selected_features, model_type, best_parameters=best_params)

    # Test final model
    final_metrics = test_final_model(final_model, df_test, y_test, dv, selected_features)
    logging.info(f'Performance of final model on test data: {final_metrics}')

    # Save model
    if save_model:
        
        output_file=f'{output_path}/{model_type}_HostRangeClassifier_{model_version}.bin'

        with open(output_file,'wb') as f_out:
            pickle.dump((dv,final_model),f_out)

        logging.info(f'Model is saved as {output_file}')

if __name__=='__main__':
    main()