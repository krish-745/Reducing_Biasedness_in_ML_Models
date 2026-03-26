import pandas as pd
import numpy as npv

class DisparateImpactRemover:
    """
    An implementation of the repairing algorithm from:
    "Certifying and Removing Disparate Impact" (Feldman et al., KDD 2015)
    """

    def __init__(self, repair_level):
        """
        repair_level: float between 0.0 and 1.0. 
            - 1.0 means full repair (feature distributions are completely identical across groups).
            - 0.0 means no repair (returns original data).
            - 0.8 means partial repair (moves values 80% of the way to the fair target distribution).
        """
        if not 0.0 <= repair_level <= 1.0:
            raise ValueError("repair_level must be between 0.0 and 1.0")
        self.repair_level = repair_level

    def fit_transform(self, df, protected_attribute, features_to_repair):
        """
        df: pandas DataFrame containing the data
        protected_attribute: string, name of the column containing the protected class (e.g., 'gender')
        features_to_repair: list of strings, continuous features to be repaired
        """
        repaired_df = df.copy()

        # Identify the distinct groups within the protected attribute
        groups = df[protected_attribute].unique()

        for feature in features_to_repair:
            # 1. Compute the percentiles (quantiles) of the feature values within each group
            group_quantiles = {}
            for group in groups:
                group_data = df[df[protected_attribute] == group][feature]
                # Rank each value as a percentage (from 0.0 to 1.0)
                group_quantiles[group] = group_data.rank(
                    pct=True, method='average')

            # 2. Reconstruct the feature mapping to a single uniform distribution
            for group in groups:
                mask = df[protected_attribute] == group
                quantiles = group_quantiles[group]

                # To achieve demographic parity, we find what value these quantiles
                # correspond to in the overall dataset distribution.
                target_values = df[feature].quantile(quantiles).values

                # Original biased values
                original_values = df.loc[mask, feature].values

                # 3. Apply the repair based on the specified repair_level budget
                repaired_values = (
                    1 - self.repair_level) * original_values + (self.repair_level * target_values)

                # Update the dataframe
                repaired_df.loc[mask, feature] = repaired_values

        return repaired_df
