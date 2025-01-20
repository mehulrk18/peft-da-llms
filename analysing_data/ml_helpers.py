import argparse
import os.path
import gc
import math
from sklearn import linear_model
from datetime import datetime
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
import xgboost as xgb
import functools
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from collections import Counter
# evaluation of a model using 10 features chosen with correlation
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error



baseline_feature_target = ['target_rouge1', 'target_rouge2', 'target_rougeL',
                          # 'target_vocab_overlap'
                           ]

baseline_feature_source = ['source_rouge1', 'source_rouge2', 'source_rougeL',
                           #'source_vocab_overlap'
                           ]

domain_specific_features = ['learning_difficult', 'vocab-overlap',
                             'kl-divergence',
                            # 'js-divergence',
                             'tf-idf-overlap',
                            'source_shannon_entropy','target_shannon_entropy'
                            ]
features_to_normalize = {'source': ['source_bert_precision', 'source_bert_recall', 'source_bert_f1', 'source_vocab_overlap',
                            'source_Relevance', 'source_Coherence', 'source_Consistency', 'source_Fluency'],
                         'all': ['source_shannon_entropy','target_shannon_entropy', 'kl-divergence', 'js-divergence',
                                 'vocab-overlap', 'tf-idf-overlap' , 'learning_difficult'] ,
                         'target': ['target_vocab_overlap', 'target_Relevance', 'target_Coherence', 'target_Consistency',
                                    'target_Fluency', 'target_bert_precision', 'target_bert_recall', 'target_bert_f1']
                         }
normalize_range = {'bert': [-1, 1], 'overlap': [0, 100], 'entropy': [0,100], 'learning_difficult': [0, 100],
                   'kl-divergence': [0, 100], 'js-divergence': [0, 100], 'Fluency': [1,3],
                   'Relevance': [1,5], 'Coherence': [1,5], 'Consistency': [1,5]

}
reduced_features_target = {
    'target_bert': ['target_bert_f1'],
    'target_rouge': ['target_rouge1', 'target_rouge2', 'target_rougeL'],
    'target_vocab_overlap_': ['target_vocab_overlap'],
    'target_llm_eval': ['target_Relevance', 'target_Coherence', 'target_Consistency', 'target_Fluency'],
    'target_fs_grounded_': ['target_fs_grounded']}
reduced_features_source = {
    'source_bert': ['source_bert_f1'],
    'source_rouge': ['source_rouge1', 'source_rouge2', 'source_rougeL'],
    'source_vocab_overlap_': ['source_vocab_overlap'],
    'source_llm_eval': ['source_Relevance', 'source_Coherence', 'source_Consistency', 'source_Fluency'],
    'source_fs_grounded_': ['source_fs_grounded']}
def xgboost(X_train, X_test, y_train, y_test ):

    # Create regression matrices
    dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
    dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)

    # Define hyperparameters
    params = {"objective": "reg:squarederror", "tree_method": "gpu_hist"}

    n = 20
    evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]

    model = xgb.train(
        params=params,
        dtrain=dtrain_reg,
        num_boost_round=n,
        evals=evals,
        verbose_eval=2
    )

    preds = model.predict(dtest_reg)
    rmse = root_mean_squared_error(y_test, preds)
    # print(f"RMSE of the base model: {rmse:.3f}")

    # Step 5: Evaluate the model's accuracy on the test set
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    #print(f"Mean Squared Error: {mse:.2f}")
    #print(f"Mean Absolute Error: {mae:.2f}")
    #print(f"R^2 Score: {r2:.2f}")
    return {'xgboost-mse': float(round(mse,3)), 'xgboost-mae': float(round(mae,3)),
            "xgboost-rmse": float(round(rmse,3)), "xgboost-r2":float(round(r2,3))}


def ridge_regression(X_train, X_test, y_train, y_test ):


    # Instantiate the Ridge Regression model
    ridge_reg = RidgeCV().fit(X_train, y_train) # You can change the alpha parameter to add more or less regularization


    # Make predictions
    y_pred = ridge_reg.predict(X_test)

    # Evaluate the model
    rmse = root_mean_squared_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    #print("Mean Squared Error (MSE):", mse)
    #print(f"Mean Absolute Error: {mae:.2f}")
    #print("R² Score:", r2)

    # Optional: Display the coefficients
    #print("Coefficients:", ridge_reg.coef_)
    #print("Intercept:", ridge_reg.intercept_)
    scores = {'ridge-mse': float(round(mse,3)), 'ridge-mae': float(round(mae,3)),
            "ridge-rmse": float(round(rmse,3)), "ridge-r2": float(round(r2,3))}
    #print(scores)
    return scores
def lasso_regression(X_train, X_test, y_train, y_test ):

    # Instantiate the Ridge Regression model
    lasso_reg = linear_model.LassoCV().fit(X_train, y_train)  # You can change the alpha parameter to add more or less regularization



    # Make predictions
    y_pred = lasso_reg.predict(X_test)

    # Evaluate the model
    rmse = root_mean_squared_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    #print("Mean Squared Error (MSE):", mse)
    #print(f"Mean Absolute Error: {mae:.2f}")
    #print("R² Score:", r2)

    # Optional: Display the coefficients
    #print("Coefficients:", lasso_reg.coef_)
    #print("Intercept:", lasso_reg.intercept_)
    scores = {'lasso-mse': float(round(mse,3)), 'lasso-mae': float(round(mae,3)),
            "lasso-rmse": float(round(rmse,3)), "lasso-r2": float(round(r2,3))}
    #print (scores)
    return scores

def linear_regression(X_train, X_test, y_train, y_test ):

    # Instantiate the Ridge Regression model
    reg = LinearRegression() # You can change the alpha parameter to add more or less regularization

    # Train the model
    reg.fit(X_train, y_train)

    # Make predictions
    y_pred = reg.predict(X_test)

    # Evaluate the model
    rmse = root_mean_squared_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    #print("Mean Squared Error (MSE):", mse)
    #print(f"Mean Absolute Error: {mae:.2f}")
    #print("R² Score:", r2)

    # Optional: Display the coefficients
    #print("Coefficients:", reg.coef_)
    #print("Intercept:", reg.intercept_)
    scores = {'mse': float(round(mse,3)), 'mae': float(round(mae,3)), "rmse": float(round(rmse,3)), "r2": float(round(r2,3))}
    #print (scores)
    return scores


def weighted_average(nums, weights):
  return sum(x * y for x, y in zip(nums, weights)) / sum(weights)
def weighted_average_list(nums, weights):
  results = []
  for num in nums:
      res = sum(x * y for x, y in zip(num, weights)) / sum(weights)
      results.append(res)
  return results


def derive_baseline_features_red(df):
    df = df[domain_specific_features + baseline_feature_source + baseline_feature_target]

    df['rouge_target'] = df[baseline_feature_target].mean(axis=1)
    df['rouge_source'] = df[baseline_feature_source].mean(axis=1)
    df = df.drop(baseline_feature_source + baseline_feature_target, axis=1)

    weighted_y_target = df['rouge_target']
    weighted_y_source = df['rouge_source']

    y_drop = np.subtract(weighted_y_source, weighted_y_target)

    df['weighted_y_target'] = weighted_y_target
    df['weighted_y_source'] = weighted_y_source
    df['y_drop'] = y_drop

    return df

def derive_baseline_features(df):
    df = df[
        domain_specific_features +
        baseline_feature_source +
        baseline_feature_target + ['y_drop']

    ]


    feature_weight = [1 / len(baseline_feature_target)] * len(baseline_feature_target)

    weighted_y_target = weighted_average_list((df[baseline_feature_target]).values, feature_weight)
    weighted_y_source = weighted_average_list(df[baseline_feature_source].values, feature_weight)

    y_drop = np.subtract(weighted_y_source, weighted_y_target)

    df['weighted_y_target'] = weighted_y_target
    df['weighted_y_source'] = weighted_y_source
    #df['y_drop'] = y_drop


    return df

def normalize_features(df):
    def norm(df, features_to_normalize, update_y = None, a = None):
        for feature in features_to_normalize:
            if feature in df.columns:
                numbers = np.array(df[feature]).reshape((-1,1))
                old_range_key = [key for key in normalize_range.keys() if key in feature][0]
                df[feature] = np.interp(numbers, normalize_range[old_range_key], [0,1])
        weighted_y_col = []
        if a is not None:
            for col in df.columns:
                if a in col:
                    weighted_y_col.append(col)

            # update weighted y
            feature_weight = [1 / len(weighted_y_col)] * len(weighted_y_col)

            df[update_y] = weighted_average_list((df[weighted_y_col]).values, feature_weight)

        return df

    #print (df.columns)
    # normalize all
    df = norm(df, features_to_normalize['all'])
    # normalize target
    df = norm(df, features_to_normalize['target'], update_y= 'weighted_y_target', a = 'target_')
    # normalize source
    df = norm(df, features_to_normalize['source'], update_y='weighted_y_source', a= 'source_')
    df['y_drop'] = df['weighted_y_source'] - df['weighted_y_target']
    return df

# feature selection
def select_features(X, y):
    feature_names = X.columns
    k = int(math.sqrt(len(X))) - 2
    if k < len(feature_names)-1:
        # configure to select a subset of features
        fs = SelectKBest(score_func=f_regression, k=k)
        # learn relationship from training data
        fs.fit(X, y)
        cols_idxs = fs.get_support(indices=True)
        features_df_new = list(X.iloc[:, cols_idxs].columns)

        # what are scores for the features
        #for i in range(len(fs.scores_)):
        #    print(f'Feature {feature_names[i]}%d: %f' % (i, fs.scores_[i]))
        # transform train input data
        X_train_fs = fs.transform(X)

        print (f"Automatic Reduced features from {len(feature_names)} to {len(X_train_fs[0])}")
    else:
        X_train_fs = X
        features_df_new = []
        print ("No automatic feature reduction performed")
    return X_train_fs, y, features_df_new

def run_regression(df:pd.DataFrame, mode:str, feature_selection_bool:bool):
    if mode == "baseline-raw" or mode == 'baseline-norm':
        print(mode)
        features_to_drop = baseline_feature_target + ['weighted_y_target','weighted_y_source',
                                                      #'source_shannon_entropy', 'js-divergence', 'vocab-overlap',
                                                      ]
    elif mode == 'all-raw' or mode == 'all-norm':
        print (mode)
        features_to_drop = ['ft','weighted_y_source', 'target_bert_f1',  'target_rouge1', 'target_rouge2',
                            'target_rougeL', 'target_vocab_overlap','target_Relevance', 'target_Coherence',
                            'target_Consistency', 'target_Fluency','da-type','source', 'target',
                            'target_fs_grounded',  'weighted_y_target',
                            'target_bert_precision', 'target_bert_recall',  'source_shannon_entropy',
                            # 'js-divergence', 'vocab-overlap',
                  ]
    elif mode == 'all-red':
        print (mode)
        features_to_drop = ['weighted_y_target', 'weighted_y_source', 'js-divergence', 'vocab-overlap','source_shannon_entropy'] + list(reduced_features_target.keys())
    elif mode == 'baseline-norm-red':
        print (mode)
        features_to_drop = ['weighted_y_target', 'weighted_y_source','js-divergence', 'vocab-overlap','source_shannon_entropy', 'rouge_target',]
    else:
        print ("mode unknown. No Regression took place.")
        return

    df = df.drop(features_to_drop, axis=1)
    df = df.dropna()
    df = df.sample(frac=1)
    # Extract feature and target arrays
    X, y = df.drop('y_drop', axis=1), df[['y_drop']]
    # Extract text features
    cats = X.select_dtypes(exclude=np.number).columns.tolist()

    # Convert to Pandas category
    for col in cats:
        X[col] = X[col].astype('category')

    # Split the dataset into training and testing sets
    selected_features = []
    if feature_selection_bool:
        X, y, selected_features = select_features(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,)
    print ("Predictions with XGBoost")
    #xgboost_scores =  xgboost(X_train, X_test, y_train, y_test)
    xgboost_scores = {'xgboost-mse': 0, 'xgboost-mae': 0, "xgboost-rmse": 0, "xgboost-r2":0}

    print("Predictions with Linear Regression")
    reg_scores = linear_regression(X_train, X_test, y_train, y_test)

    print ("Predictions with Ridge Regression")
    ridge_scores = ridge_regression(X_train, X_test, y_train, y_test)

    print ("Predictions with Lasso Regression")
    lasso_scores = lasso_regression(X_train, X_test, y_train, y_test)

    ridge_scores.update(reg_scores)
    ridge_scores.update(xgboost_scores)
    ridge_scores.update(lasso_scores)
    feature_score = {'features':mode, 'feature_selection': feature_selection_bool}
    feature_score.update(ridge_scores)

    return feature_score, selected_features

def reduce_features_all(df):
    len_before =  len(df.columns)
    reduced_features_target_values = [v for value in reduced_features_target.values() for v in value]
    reduced_features_source_values = [v for value in reduced_features_source.values() for v in value]


    df = df[domain_specific_features + reduced_features_target_values + reduced_features_source_values]

    for key, value in reduced_features_target.items():
        df[key] = df[value].mean(axis = 1)

    for key, value in reduced_features_source.items():
        df[key] = df[value].mean(axis = 1)

    df = df.drop(reduced_features_target_values + reduced_features_source_values, axis = 1)
    num_features = len(reduced_features_target.keys())
    feature_weight = [1 / num_features] * num_features

    weighted_y_target = weighted_average_list((df[list(reduced_features_target.keys())]).values, feature_weight)
    weighted_y_source = weighted_average_list(df[list(reduced_features_source.keys())].values, feature_weight)

    y_drop = np.subtract(weighted_y_source, weighted_y_target)

    df['weighted_y_target'] = weighted_y_target
    df['weighted_y_source'] = weighted_y_source
    df['y_drop'] = y_drop

    len_after= len(df.columns)
    print (f"Manually Reduced features from {len_before} to {len_after}")
    return df

def clear_cache():
    gc.collect()
    objects = [i for i in gc.get_objects()
               if isinstance(i, functools._lru_cache_wrapper)]

    # All objects cleared
    for object in objects:
        object.cache_clear()