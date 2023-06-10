from sklearn import metrics
import numpy as np


from scipy import stats


def eval_model(labels, predictions):    
    accuracy = metrics.accuracy_score(y_true=labels, y_pred=(predictions > 0.5).astype(int))
    try:
        auroc = metrics.roc_auc_score(labels, predictions)
    except:
        auroc = 0
    auprc = metrics.average_precision_score(labels, predictions)
    logloss = metrics.log_loss(labels, predictions, labels=[0, 1])
    
    # ci of accuracy
    correct = (np.array(predictions) > 0.5).astype(int) == labels
    
    ci_l, ci_u = stats.bootstrap((correct,), statistic=np.mean, confidence_level=0.95, n_resamples=1000, method='percentile').confidence_interval
    
    res = {
        "auroc": auroc, "auprc": auprc, "accuracy": accuracy,
        "avg": np.mean(predictions), "meanlabel": np.mean(labels),
        "accuracy_lower_95ci": ci_l,
        "accuracy_upper_95ci": ci_u,
        "logloss": logloss,
    }
    return res



def eval_sample(sample):
    sample = np.array(sample)
    labels = sample[:, 1]
    predictions = sample[:, 0]
    return metrics.accuracy_score(y_true=labels, y_pred=(predictions > 0.5).astype(int))
    
    
def report_accuracy_with_std_dev(all_states, big_states, years=[2019, 2021], split='validation'):
    metric = 'accuracy'
    for state in big_states:
        print(state)
        for method in ['all', 'aux', 'without']:
            r2019 = all_states[state][2019][split][method][metric]
            r2019_l = all_states[state][2019][split][method]['accuracy_std']
            r2021 = all_states[state][2021][split][method][metric]
            r2021_l = all_states[state][2021][split][method]['accuracy_std']
            
            print(f"2019 {r2019:.3f} +/- {r2019_l:.4f}, 2021 {r2021:.3f}  +/-  {r2021_l:.4f}  for {method}")

    m = {
        2019: {'all': [], 'aux': [], 'without':[]},
        2021: {'all': [], 'aux': [], 'without':[]}
    }

    for state in big_states:
        
        results = f"{state} "
        for year in years:
            for method in ['all', 'aux', 'without']:
                r = all_states[state][year][split][method][metric]
                r_std = all_states[state][year][split][method]['accuracy_std']
                results = results + f" & {round(r,3)} \plusminus {round(r_std,4)}"
                
                m[year][method].append(r)
                
        print(f"{results} \\\\")
    for year in years:
        print(year)
        for method in ['all', 'aux', 'without']:  
            print(method)
            print(
             f"mean {round(np.mean(m[year][method]),3)}  min {np.min(m[year][method]):.3f} q25 {np.quantile(m[year][method], 0.25):.3f} q50 {np.quantile(m[year][method], 0.5):.3f} q75 {np.quantile(m[year][method], 0.75):.3f}")
    

def report_accuracy(all_states, big_states, years=[2019, 2021], split='validation'):
    metric = 'accuracy'
    for state in big_states:
        print(state)
        for method in ['all', 'aux', 'without']:
            r2019 = all_states[state][2019][split][method][metric]
            r2019_l = all_states[state][2019][split][method]['accuracy_lower_95ci']
            r2019_u = all_states[state][2019][split][method]['accuracy_upper_95ci']
            r2021 = all_states[state][2021][split][method][metric]
            r2021_l = all_states[state][2021][split][method]['accuracy_lower_95ci']
            r2021_u = all_states[state][2021][split][method]['accuracy_upper_95ci']
            
            print(f"2019 {r2019:.3f} [{r2019_l:.4f}, {r2019_u:.4f}] 2021 {r2021:.3f} [{r2021_l:.4f}, {r2021_u:.4f}] for {method}")

    m = {
        2019: {'all': [], 'aux': [], 'without':[]},
        2021: {'all': [], 'aux': [], 'without':[]}
    }

    
    for state in big_states:
        results = f"{state} "
        for year in years:
            for method in ['all', 'aux', 'without']:
                r = all_states[state][year][split][method][metric]
                r_l = all_states[state][year][split][method]['accuracy_lower_95ci']
                r_u = all_states[state][year][split][method]['accuracy_upper_95ci']
                results = results + f" & {round(r,3)} [{round(r_l,3)}, {round(r_u,3)}]"

                m[year][method].append(r)

        print(f"{results} \\\\")

    for year in years:
        print(year)
        for method in ['all', 'aux', 'without']:  
            print(method)
            print(
             f"mean {round(np.mean(m[year][method]),3)}  min {np.min(m[year][method]):.3f} q25 {np.quantile(m[year][method], 0.25):.3f} q50 {np.quantile(m[year][method], 0.5):.3f} q75 {np.quantile(m[year][method], 0.75):.3f}")