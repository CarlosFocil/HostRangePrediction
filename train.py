import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split,cross_validate
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import make_scorer, roc_auc_score

from sklearn.tree import DecisionTreeClassifier

import pickle

# PARAMETERS

mi_threshold=0.12
max_depth = None
min_samples_leaf=5

print('Starting script for training a Decision Tree with the following parameters:')
print(f'Mutual information threshold:{mi_threshold}, max_depth={max_depth},min_samples_leaf = {min_samples_leaf}')

output_file = f'model_mi={mi_threshold}_depth={max_depth}_min_samples_leaf={min_samples_leaf}.bin'

# DATA LOADING AND PREPROCESSING
## Aerobic data
print('Loading and pre-processing intial data')
data_aerobic = pd.read_excel('raw_data/seif_support_catabolic-data.xlsx', sheet_name=2, header=2, index_col=0)
data_aerobic.columns = data_aerobic.columns.str.replace(' ','_').str.lower()
data_aerobic = data_aerobic.add_suffix('_ae')

good_columns = data_aerobic.sum()[data_aerobic.sum().round(5) > 0.01].index.to_list()
clean_aerobic = data_aerobic[good_columns]

## Anaerobic data
data_ana = pd.read_excel('raw_data/seif_support_catabolic-data.xlsx', sheet_name=3, header=2, index_col=0)
data_ana.columns = data_ana.columns.str.replace(' ','_').str.lower()
data_ana = data_ana.add_suffix('_ana')

good_columns = data_ana.sum()[data_ana.sum().round(5) > 0.01].index.to_list()
clean_ana = data_ana[good_columns]

clean_aerobic = clean_aerobic.reset_index(names='strain_id')
clean_ana = clean_ana.reset_index(names='strain_id')

## Joining dataframes
joined_df = pd.merge(clean_aerobic, clean_ana, on ='strain_id')

## Load metadata
metadata = pd.read_excel('raw_data/seif_metadata.xlsx', sheet_name=0, header=1)
metadata.columns = metadata.columns.str.lower().str.replace(' ','_')
columns = ['genome_accession', 'host_range']
df2 = metadata[columns]

## Join dataframes
full_df = pd.merge(joined_df, df2, left_on='strain_id',right_on='genome_accession')
full_df = full_df.loc[full_df.host_range.dropna().index].reset_index(drop=True)

del full_df['strain_id']
del full_df['genome_accession']

numeric = full_df.columns.to_list()
numeric.remove('host_range')
full_df[numeric] = full_df[numeric].clip(lower=0)

encoded = (full_df.host_range == 'Broad').astype(int)
encoded.name = 'host_range_encoded'
full_df = pd.concat([full_df, encoded], axis=1)

# FEATURE SELECTION

print('Selecting features based on mutual information threshold')
# Separate the features and the target variable
X = full_df.drop(columns=['host_range', 'host_range_encoded'])
y = full_df['host_range_encoded']
# Calculate mutual information and saving the results as a dataframe
mutual_info = mutual_info_classif(X, y)
mi_df = pd.DataFrame({'feature': X.columns, 'mutual_information': mutual_info})
mi_df.sort_values(by='mutual_information', ascending=False, inplace=True)
selected_features = mi_df[mi_df.mutual_information >= mi_threshold].feature.to_list()


# SPLIT
df_full_train, df_test = train_test_split(full_df, test_size=0.2, random_state=11)

df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_full_train = df_full_train.host_range_encoded.values
y_test = df_test.host_range_encoded.values

del df_full_train['host_range']
del df_full_train['host_range_encoded']

del df_test['host_range']
del df_test['host_range_encoded']

# Train model
def model_validation(max_depth,min_samples_leaf,selected_features, cv=5):
    print(f'Validating decision tree with max_depth: {max_depth}, min_samples_leaf: {min_samples_leaf} and {cv} folds')
    
    # Vectorize dicts and prepare matrices
    full_train_dicts = df_full_train[selected_features].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_full_train = dv.fit_transform(full_train_dicts)

    # Define decision tree classifier model
    dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=11)
    cv_results = cross_validate(dt, X_full_train, y_full_train, cv=cv, scoring={'roc_auc': make_scorer(roc_auc_score, needs_proba=True)})

    mean_roc_auc = np.mean(cv_results['test_roc_auc'])
    std_roc_auc = np.std(cv_results['test_roc_auc'])

    print(f'Cross-validation results: Mean = {mean_roc_auc}, std_dev = {std_roc_auc}')

model_validation(max_depth=max_depth, min_samples_leaf=min_samples_leaf, selected_features=selected_features,cv=5)

def train(df_train, y_train, max_depth, min_samples_leaf, selected_features):
    dicts = df_train[selected_features].to_dict(orient='records')

    dv=DictVectorizer(sparse=False)
    X_train=dv.fit_transform(dicts)

    model = DecisionTreeClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,random_state=11)
    model.fit(X_train,y_train)
    
    return dv,model

def predict(df_test,dv,model):
    dicts=df_test[selected_features].to_dict(orient='records')

    X_test = dv.transform(dicts)
    y_pred = model.predict_proba(X_test)[:,1]

    return y_pred

print('Training final model')
dv,model = train(df_full_train, y_train=y_full_train, max_depth=max_depth, min_samples_leaf=min_samples_leaf, selected_features=selected_features)
y_pred = predict(df_test=df_test,dv=dv,model=model)

test_roc_auc = roc_auc_score(y_test,y_pred)
print(f'auc={test_roc_auc}')

with open(output_file,'wb') as f_out:
    pickle.dump((dv,model),f_out)

print(f'Model is saved as {output_file}')