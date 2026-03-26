import numpy as np
import pandas as pd

def reweighing(df, prot, target):
    df_out = df.copy()
    df_out['weight'] = 1.0
    
    p=df_out[target].mean()
    
    groups=df_out.groupby(prot)
    
    for keys,group in groups:
        p_g=len(group)/len(df_out)
        
        for val,p_target in [(1,p),(0,1-p)]:
            mask=(group[target]==val)
            p__=mask.sum()/len(df_out)
            ep__=p_g*p_target
            
            if p__ > 0:
                weight=ep__/p__
                df_out.loc[group.index[mask],'weight']=weight
    return df_out


    