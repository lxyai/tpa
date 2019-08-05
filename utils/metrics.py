import numpy as np


def mean_mape(preds, labels):
    return np.mean(np.abs((preds - labels) / labels))


def mape_and_compliance_rate(preds, labels):
    labels[labels == 0] = 1
    all_mape = np.abs((preds - labels) / labels)
    final_mape = np.mean(all_mape[labels != 0])
    compliance_rate = np.count_nonzero(all_mape < 0.1) / np.count_nonzero(labels != 0)
    return final_mape, compliance_rate


def mae(preds, labels, scale):
    all_mae = np.abs(preds - labels)
    print(all_mae)
    final_mae = np.mean(all_mae)
    return final_mae

def rrse(preds, labels, mean):
    np.sqrt(np.sum((preds-labels)**2)) / np.sqrt((preds - mean)**2)
