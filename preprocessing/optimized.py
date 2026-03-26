import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import cvxpy as cp

class OptimizedPreprocessor:
    def __init__(self,prot,target,e=1e-5,distortion=1):
        self.prot=prot if isinstance(prot, list) else [prot]
        self.target=target
        self.e=e
        self.distortion=distortion

        self.states={}
        self.inverse_states={}
        self.true=[]
        self.n=0
        
        self.d=[]
        self.groups={}
        self.M={}
        self.distortion_matrix=None
    
    def discretize(self,df,cols,bins=4):
        df_=df.copy()
        for col in cols:
            df_[col]=pd.qcut(df_[col],bins,labels=False,duplicates='drop')
        return df_
    
    def fit(self,df,cols=None,bins=4):
        if cols is None: cols=[]
        
        self.bins=bins
        
        print("1. Discretizing continuous data and mapping state space...")
        df_=self.discretize(df,cols,bins)
        
        x=[c for c in df_.columns if c not in self.prot]
        combinations=df_[x].drop_duplicates().values
        self.n=len(combinations)
        
        target_idx=x.index(self.target)
        
        for state_id,combination in enumerate(combinations):
            self.states[tuple(combination)]=state_id
            self.inverse_states[state_id]=tuple(combination)
            if combination[target_idx]==1:
                self.true.append(state_id)
                
        print(f"   -> Found {self.n} unique applicant archetypes (states).")
        
        print("2. Computing individual distortion matrix (Fast Vectorization)...")
        state_array = np.array([self.inverse_states[i] for i in range(self.n)])
        self.distortion_matrix = cdist(state_array, state_array, metric='cityblock')
        
        target_vals = state_array[:, target_idx]
        penalty_mask = (target_vals[:, None] == 1) & (target_vals[None, :] == 0)
        self.distortion_matrix += penalty_mask * 10.0
                
        print("3. Extracting intersectional protected groups...")
        self.d = [tuple(x) for x in df_[self.prot].drop_duplicates().values]
        print(f"   -> Tracking {len(self.d)} distinct protected groups.")
        
        for group in self.d:
            dist=np.zeros(self.n)
            condition=(df_[self.prot]==group).all(axis=1)
            df_group=df_[condition]
            
            for i,row in df_group.iterrows():
                state_id=self.states[tuple(row[x])]
                dist[state_id]+=1
            
            self.groups[group]=dist/len(df_group) if len(df_group)>0 else dist
            
        print("4. Running CVXPY Convex Optimization Engine...")
        self.optimize()
        return self
    
    def optimize(self):
        constraints = []
        loss = 0
        new_distributions = {}
        true_states = {}
        
        true_mask = np.zeros(self.n)
        for idx in self.true:
            true_mask[idx] = 1.0
            
        for group in self.d:
            matrix = cp.Variable((self.n, self.n), nonneg=True)
            self.M[group] = matrix
            dist = self.groups[group]
            
            new_dist = matrix.T @ dist
            new_distributions[group] = new_dist
            
            constraints.append(cp.sum(matrix, axis=1) == 1)
            
            expected_distortions = cp.sum(cp.multiply(matrix, self.distortion_matrix), axis=1)
            constraints.append(expected_distortions <= self.distortion)
                
            loss += cp.sum(cp.abs(new_dist - dist))
            
            rate = true_mask @ new_dist
            true_states[group] = rate
            
        for i, g1 in enumerate(self.d):
            for j in range(i+1, len(self.d)):
                g2 = self.d[j]
                constraints.append(cp.abs(true_states[g1] - true_states[g2]) <= self.e)
                
        problem = cp.Problem(cp.Minimize(loss), constraints)
        print("   -> Compiling and Solving (Vectorized)...")
        problem.solve() 
        
        if problem.status in ["infeasible", "unbounded"]:
            raise ValueError(
                f"Optimization Infeasible! The constraints are too strict. "
                f"Try increasing epsilon (currently {self.e}) or "
                f"distortion_budget (currently {self.distortion})."
            )
        
        print(f"   -> Optimization Successful! Minimal Utility Loss: {problem.value:.4f}")
        
        for group in self.d:
            self.M[group] = self.M[group].value
            
    def transform(self,df,cols=None):
        if not self.M:
            raise Exception("You must call .fit() before .transform()")
        
        if cols is None: cols = []
        
        print("5. Transforming dataset based on optimal matrices...")
        
        df_=self.discretize(df,cols,self.bins)
        x=[c for c in df_.columns if c not in self.prot]
        
        transformed_rows=[]
        
        for i,row in df_.iterrows():
            current_state=self.states[tuple(row[x])]
            group=tuple(row[self.prot])
            
            matrix=self.M[group]
            p=matrix[current_state,:]
            
            p = np.clip(p, 0, 1)
            if p.sum() == 0:
                p[current_state] = 1.0 
            p /= p.sum()

            new_state=np.random.choice(self.n,p=p)
            new_x=self.inverse_states[new_state]
            
            new_row=dict(zip(x,new_x))
            for attr,val in zip(self.prot,group):
                new_row[attr]=val
            transformed_rows.append(new_row)
            
        print("   -> Transformation complete.")
        return pd.DataFrame(transformed_rows)
        
    
    
        
        
        
        