import numpy as np


def mean_mape(preds, labels):
    return np.mean(np.abs((preds - labels) / labels))


def mape_and_compliance_rate(preds, labels):
    all_mape = np.abs((preds - labels) / labels)
    final_mape = np.mean(all_mape)
    compliance_rate = np.count_nonzero(all_mape < 0.1) / np.prod(labels.shape)
    return final_mape, compliance_rate
