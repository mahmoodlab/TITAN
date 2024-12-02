
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import log_loss
from sklearn.preprocessing import Normalizer, StandardScaler
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from titan.utils import get_eval_metrics, seed_torch


def train_and_evaluate_logistic_regression_with_val(train_data, train_labels, val_data, val_labels, test_data, test_labels, log_spaced_values=None, max_iter=500):
    seed_torch(torch.device('cpu'), 0)
    
    metric_dict = {
        'bacc': 'balanced_accuracy',
        'kappa': 'cohen_kappa_score',
        'auroc': 'roc_auc_score',
    }
    
    # Logarithmically spaced values for regularization
    if log_spaced_values is None:
        log_spaced_values = np.logspace(np.log10(10e-6), np.log10(10e5), num=45)
    
    # loop over log_spaced_values to find the best C
    best_score = -float('inf')
    best_C = None
    logistic_reg_final = None
    for log2_coeff in tqdm(log_spaced_values, desc="Finding best C"):
        # suppress convergence warnings
        import warnings
        warnings.filterwarnings("ignore")
        
        logistic_reg = LogisticRegression(
            C=1/log2_coeff,
            fit_intercept=True,
            max_iter=max_iter,
            random_state=0,
            solver="lbfgs",
        )
        logistic_reg.fit(train_data, train_labels)
        
        # predict on val set
        val_loss = log_loss(val_labels, logistic_reg.predict_proba(val_data))
        score = -val_loss
        
        # score on val set
        if score > best_score:
            best_score = score
            best_C = log2_coeff
            logistic_reg_final = logistic_reg
    print(f"Best C: {best_C}")
    
    # Evaluate the model on the test data
    test_preds = logistic_reg_final.predict(test_data)
    
    num_classes = len(np.unique(train_labels))
    if num_classes == 2:
        test_probs = logistic_reg_final.predict_proba(test_data)[:, 1]
        roc_kwargs = {}
    else:
        test_probs = logistic_reg_final.predict_proba(test_data)
        roc_kwargs = {'multi_class': 'ovo', 'average': 'macro'}

    eval_metrics = get_eval_metrics(test_labels, test_preds, test_probs, roc_kwargs=roc_kwargs)
    
    outputs = {
        "targets": test_labels,
        "preds": test_preds,
        "probs": test_probs,
    }
        
    return eval_metrics, outputs