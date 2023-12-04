import numpy as np
import os
from models import cross_validation

vname = lambda v,nms: [ vn for vn in nms if id(v)==id(nms[vn])][0]


def combine_feature(feature1: np.ndarray or None, feature2: np.ndarray):
    if feature1 is None:
        return feature2

    N1, H1, W1 = feature1.shape
    N2, H2, W2 = feature2.shape

    assert N1 == N2

    N = N1
    H = H1 + H2
    W = max(W1, W2)
    feature = np.zeros((N, H, W), dtype = np.float32)
    feature[:, 0: H1, 0: W1] = feature1
    feature[:, H1: H, 0: W2] = feature2

    return feature


class ForwardSelection:

    def __init__(self, features, labels, flags, names, losses, accuracies, aucs, out_dir):
        self.flags = flags
        self.features = features
        self.labels = labels
        self.names = names
        self.losses = losses
        self.accuracies = accuracies
        self.aucs = aucs
        self.out_dir = out_dir

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

    def forward_selection(self, metric: str, feature = None, selected = np.empty((1, 0))):
        """
        @param metric: "losses" or "accuracies" or "aucs"
        """

        assert metric == "losses" or metric == "accuracies" or metric == "aucs"

        count = np.sum(self.flags != 0)
        if count == self.flags.shape[0]:
            return selected

        for i in np.where(self.flags == 0)[0]:
            new_feature = combine_feature(feature, self.features[i])

            if new_feature.shape[1] < 6:
                continue

            new_selected = np.append(selected, self.names[i])

            print(f"Selected feature: {new_selected}")

            out_subdir = os.path.join(self.out_dir, "_".join(new_selected))
            n_epoch = 20
            acc, loss, auc = cross_validation(new_feature, self.labels, n_epoch, out_subdir)

            # Save records
            self.losses[count, i] = loss
            self.accuracies[count, i] = acc
            self.aucs[count, i] = auc
        
        metric_matrix = eval(f"self.{metric}")
        if metric == "accuracies" or metric == "aucs":
            metric_matrix = -metric_matrix
            
        if count > 0 and min(metric_matrix[count, :]) >= min(metric_matrix[count - 1, :]):
            return selected

        i_min = np.argmin(metric_matrix[count, :])

        new_feature = combine_feature(feature, self.features[i_min])
        new_selected = np.append(selected, self.names[i_min])
        self.flags[i_min] = count + 1

        res = self.forward_selection(metric, new_feature, new_selected)

        return res