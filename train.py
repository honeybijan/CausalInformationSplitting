import numpy as np

import folktables


from evaluate import eval_model


from sklearn.linear_model import LinearRegression, LogisticRegression


def make_prediction_problem(stable_features, unstable_features):
    return folktables.BasicProblem(
        features=stable_features + unstable_features,
        target='PINCP',
        target_transform=lambda x: x > 50000,
        preprocess=folktables.acs.adult_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )



def make_x(x, columns):
    copy_x = x.copy()
    for c in columns:  # pad missing columns
        if c not in x.columns:
            copy_x[c] = 0
    return copy_x[columns] # eliminate extra columns

def add_noise(x, feature):
    copy_x = x.copy()
    noise = np.random.normal(-7, 7, x.shape[0]) 
    copy_x[feature] = copy_x[feature] + noise
    return copy_x
    

def make_transform(aux_predictors, unstable_features, stable_features):

    def transform(x):
        
        x_stable = make_x(x, stable_features)
        
        input_aux = make_x(x, unstable_features)
        aux_predictions = []
        for predictor in aux_predictors:
            aux_predictions.append(np.expand_dims(predictor.predict(make_x(x, unstable_features)), axis=1))
        
        res = np.concatenate(
            [make_x(x, stable_features)] + aux_predictions, axis=1)
        return res
    return transform


def train(by_year_data_split, categories, year, stable_feature_names, unstable_feature_names, unstable_feature_targets, penalty='l1', solver='liblinear'):
    ACSIncomeAllFeature = make_prediction_problem(stable_feature_names, unstable_feature_names)
    
    x, y, _ = ACSIncomeAllFeature.df_to_pandas(by_year_data_split["train"][year], categories=categories, dummies=True)
    y = y.values[:, 0]
    feature_names = x.columns
    stable_features = []
    unstable_features = []
    for f in feature_names:
        for t in stable_feature_names:
            if f.startswith(t):
                stable_features.append(f)
        for t in unstable_feature_names:
            if f.startswith(t):
                unstable_features.append(f)
    
    print(feature_names)
    print("stable")
    print(stable_feature_names)
    print(stable_features)
    print("unstable")
    print(unstable_feature_names)
    print(unstable_features)
    
    # all
    model_all = LogisticRegression(penalty=penalty, solver=solver)
    model_all.fit(make_x(x, stable_features + unstable_features), y)
    # without
    model_without = LogisticRegression(penalty=penalty, solver=solver)
    model_without.fit(make_x(x, stable_features), y)
    # aux
    aux_predictors = []
    for unstable_feature_target, model in unstable_feature_targets:
        print(list(feature_names))
        print(unstable_feature_target)
        assert unstable_feature_target in list(feature_names)  # make sure the comparison to all / without makes sense
        x_aux = make_x(x, unstable_features + [unstable_feature_target]).values
        x_true = x_aux[y, :]
        x_false = x_aux[np.logical_not(y), :]

        clf_aux_true = model().fit(x_true[:, :-1], x_true[:, -1])
        clf_aux_false = model().fit(x_false[:, :-1], x_false[:, -1])
        aux_predictors += [clf_aux_true, clf_aux_false]
    aux_transform = make_transform(aux_predictors, unstable_features, stable_features)
    model_aux = LogisticRegression(penalty=penalty, solver=solver).fit(aux_transform(x), y)
    
    metrics = {}
    for other_year in [2019, 2021]:
        metrics[other_year] = {}
        for split in ["train", "validation", "test"]:
            metrics[other_year][split] = {}
            eval_x, eval_y, _ = ACSIncomeAllFeature.df_to_pandas(
                by_year_data_split[split][other_year], categories=categories, dummies=True)
            eval_y = eval_y.values[:, 0]
            
            all_eval = eval_model(labels=eval_y, predictions=model_all.predict(make_x(eval_x, stable_features + unstable_features)))
            without_eval = eval_model(labels=eval_y, predictions=model_without.predict(make_x(eval_x, stable_features)))
            aux_eval = eval_model(labels=eval_y, predictions=model_aux.predict(aux_transform(eval_x)))
            
            # noisy
#             all_eval_noisy = eval_model(labels=eval_y, predictions=model_all.predict(add_noise(make_x(eval_x, stable_features + unstable_features), unstable_features[0])))
#             aux_eval_noisy = eval_model(labels=eval_y, predictions=model_aux.predict(aux_transform(add_noise(eval_x, unstable_features[0]))))
            
            metrics[other_year][split] = {
                "all": all_eval, "without": without_eval, "aux": aux_eval,
#                 "all_noisy": all_eval_noisy, "aux_noisy": aux_eval_noisy
            }
    return metrics