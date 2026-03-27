import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix

class FairnessAuditor:
    def __init__(self, y_true, y_pred, protected_attr, priv_class=1, unpriv_class=0, X_features=None):
        """
        Initializes the auditor.
        y_true: numpy array of actual labels (0 or 1)
        y_pred: numpy array of predicted labels (0 or 1)
        protected_attr: numpy array of the protected attribute (e.g., Gender)
        priv_class: the value in protected_attr representing the privileged group
        unpriv_class: the value in protected_attr representing the unprivileged group
        X_features: The feature matrix (required only for Consistency Score)
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.protected_attr = np.array(protected_attr)
        self.priv_class = priv_class
        self.unpriv_class = unpriv_class
        self.X = X_features

        # Masks for group filtering
        self.priv_mask = (self.protected_attr == self.priv_class)
        self.unpriv_mask = (self.protected_attr == self.unpriv_class)

    def _get_rates(self, mask):
        """Helper to calculate TPR, FPR, PPV, and Selection Rate for a specific group."""
        y_t = self.y_true[mask]
        y_p = self.y_pred[mask]
        
        # Avoid division by zero issues
        if len(y_t) == 0:
            return 0, 0, 0, 0

        tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # True Positive Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0 # False Positive Rate
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0 # Precision
        sr = np.mean(y_p)                              # Selection Rate

        return tpr, fpr, ppv, sr

    def evaluate_group_fairness(self):
        _, _, _, sr_p = self._get_rates(self.priv_mask)
        _, _, _, sr_u = self._get_rates(self.unpriv_mask)

        # Statistical Parity Difference: P(Y=1|U) - P(Y=1|P)
        spd = sr_u - sr_p
        
        # Disparate Impact: P(Y=1|U) / P(Y=1|P)
        di = sr_u / sr_p if sr_p > 0 else 0.0

        return {"SPD": spd, "DI": di}

    def evaluate_classification_fairness(self):
        tpr_p, fpr_p, ppv_p, _ = self._get_rates(self.priv_mask)
        tpr_u, fpr_u, ppv_u, _ = self._get_rates(self.unpriv_mask)

        # Equal Opportunity Difference: TPR_U - TPR_P
        eod = tpr_u - tpr_p
        
        # Average Odds Difference: Average of FPR diff and TPR diff
        aod = 0.5 * ((fpr_u - fpr_p) + (tpr_u - tpr_p))
        
        # Predictive Parity Difference: PPV_U - PPV_P
        ppd = ppv_u - ppv_p

        return {"EOD": eod, "AOD": aod, "Predictive_Parity": ppd}

    def evaluate_individual_fairness(self, n_neighbors=5):
        # 1. Theil Index (Entropy-based benefit distribution)
        # Add a small epsilon to avoid log(0)
        epsilon = 1e-10
        b_i = self.y_pred + epsilon 
        mu = np.mean(b_i)
        
        if mu == 0:
            theil_index = 0
        else:
            theil_index = np.mean((b_i / mu) * np.log(b_i / mu))

        # 2. Consistency Score (k-Nearest Neighbors)
        consistency = 1.0
        if self.X is not None:
            nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
            nn.fit(self.X)
            distances, indices = nn.kneighbors(self.X)
            
            # Compare prediction of i with predictions of its k neighbors
            inconsistency = 0.0
            for i in range(len(self.X)):
                inconsistency += np.abs(self.y_pred[i] - np.mean(self.y_pred[indices[i]]))
            consistency = 1.0 - (inconsistency / len(self.X))

        return {"Theil_Index": theil_index, "Consistency": consistency}

    def calculate_afs(self):
        """
        Normalizes all metrics to a 0-100 scale where 100 is perfectly fair, 
        and averages them for the Aggregated Fairness Score (AFS).
        """
        group = self.evaluate_group_fairness()
        cls_fair = self.evaluate_classification_fairness()
        indv = self.evaluate_individual_fairness()

        # Normalization Logic (0 to 100 scales)
        # SPD, EOD, AOD: Ideal is 0. Scale based on absolute distance from 0.
        score_spd = max(0, 100 * (1 - abs(group["SPD"])))
        score_eod = max(0, 100 * (1 - abs(cls_fair["EOD"])))
        score_aod = max(0, 100 * (1 - abs(cls_fair["AOD"])))
        score_ppd = max(0, 100 * (1 - abs(cls_fair["Predictive_Parity"])))

        # Disparate Impact: Ideal is 1. If > 1, invert it. 
        di_ratio = group["DI"] if group["DI"] <= 1 else (1 / group["DI"] if group["DI"] > 0 else 0)
        score_di = 100 * di_ratio

        # Theil Index: Ideal is 0.
        score_theil = max(0, 100 * (1 - indv["Theil_Index"]))

        # Consistency: Ideal is 1.
        score_consistency = 100 * indv["Consistency"]

        # Aggregate the scores
        all_scores = [score_spd, score_eod, score_aod, score_ppd, score_di, score_theil, score_consistency]
        afs = np.mean(all_scores)

        report = {
            "Metrics": {**group, **cls_fair, **indv},
            "Normalized_Scores": {
                "SPD_Score": score_spd,
                "DI_Score": score_di,
                "EOD_Score": score_eod,
                "AOD_Score": score_aod,
                "Predictive_Parity_Score": score_ppd,
                "Theil_Score": score_theil,
                "Consistency_Score": score_consistency
            },
            "Aggregated_Fairness_Score": afs,
            "Verdict": "Pass" if afs > 80 else ("Warning" if afs > 65 else "Fail")
        }
        
        return report