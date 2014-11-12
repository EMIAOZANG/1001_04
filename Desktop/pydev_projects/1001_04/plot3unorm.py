'''
Created on Nov 12, 2014

@author: luchristopher
'''
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cs = [10**x for x in range(-9,2)]

aucs_df = pd.read_table('res.txt',dtype = np.float64,sep='\s+')
aucs_df.columns = aucs_df.columns.astype(np.float64)
# print aucs_df
# print aucs_df.columns
aucs_mean = aucs_df.mean(axis=0)
print 'auc_mean:\n',aucs_mean
# print >> f, 'aucs_mean:\n',aucs_mean
aucs_stderr = aucs_df.sem()
#     print >> f, 'stderr(AUC):\n',np.std(aucs_df)
aucs_lower_bound = aucs_mean-aucs_stderr
aucs_upper_bound = aucs_mean+aucs_stderr
aucs_array = aucs_df.values
#     print >> f, 'array:\n',aucs_array
aucs_ref = pd.Series(np.max(aucs_array-np.std(aucs_array)),index=cs)
print aucs_ref
#     aucs_ref = pd.Series(0.7305,index=cs)
#     print >> f, 'ref:\n',aucs_ref
aucs_mean.plot(logx=True,legend=True,label='Mean',title='X-Validation on different C values')
aucs_lower_bound.plot(logx=True,style='k+',legend=True,label='Mean-Stderr')
aucs_upper_bound.plot(logx=True,style='k--',legend=True,label='Mean+Stderr')
aucs_ref.plot(logx=True,style='r',legend=True,label='Reference Line')
plt.show()   