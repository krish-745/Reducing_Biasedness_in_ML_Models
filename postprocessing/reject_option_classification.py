import pandas as pd
import numpy as np

def reject_option_classification(df,target,prot,normal,threshold=0.5,margin=0.5):
    df_out=df.copy()
    
    df_out['original_decision'] = (df_out[target] >= threshold).astype(int)
    df_out['fair_decision'] = df_out['original_decision']
    
    low=threshold-margin
    high=threshold+margin
    
    not_safe=(df_out[target] >= low) & (df_out[target] <= high)
    
    df_out.loc[not_safe & normal, 'fair_decision'] = 1
    df_out.loc[not_safe & prot, 'fair_decision'] = 0
    
    inversions = (df_out['original_decision'] != df_out['fair_decision']).sum()
    print(f"ROC applied: Flipped {inversions} uncertain decisions to achieve fairness.")
    
    return df_out.drop(columns=['original_decision'])   
