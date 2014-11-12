import pandas as pd
import numpy as np
import sys
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm, linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import *
from sklearn.metrics import *
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

def cleanBosonData(file_name):
    #read file <file_name>
    try:
        raw_data = pd.read_csv(file_name)
    except:
        print >> sys.stderr, 'sth is going wrong with the reading process!\n'
    #count values in 'Label' and create the 'Y' column as the boolean indicator
    counter = raw_data['Label'].groupby(raw_data['Label']).count().keys()[0]
    raw_data['Y'] = (raw_data['Label']!=counter).astype(int)  
    raw_data = raw_data.drop('Label',1)
    #create dummy variables
    missing_exist = (raw_data==-999).any(axis=0)
    column_list = missing_exist[(missing_exist == True)].keys()
    for columns in column_list:
        raw_data[columns+'mv'] = (raw_data[columns]==-999).astype(int)
    #replace missing values with columnwise averages
    new_data = raw_data.replace(to_replace=-999,value=np.nan,inplace=False)
    raw_data.replace(to_replace=-999,value=new_data.mean(),inplace = True)
    return raw_data

# def plotAUC(truth, pred, lab):
#     fpr, tpr, thresholds = roc_curve(truth, pred)
#     roc_auc = auc(fpr, tpr)
#     c = (np.random.rand(), np.random.rand(), np.random.rand())
# #     plt.plot(fpr, tpr, color=c, label= lab+' (AUC = %0.2f)' % roc_auc)
# #     plt.plot([0, 1], [0, 1], 'k--')
# #     plt.xlim([0.0, 1.0])
# #     plt.ylim([0.0, 1.0])
# #     plt.xlabel('FPR')
# #     plt.ylabel('TPR')
# #     plt.title('ROC')
# #     plt.legend(loc="lower right")
#     plt.plot(tpr, thresholds, color=c, label= lab+' (AUC = %0.2f)' % roc_auc)
#     plt.plot([0,1],[0,1],'k--')
#     plt.xlim([0.,1.])
#     plt.ylim([0.,1.])
#     plt.xlabel('TPR')
#     plt.ylabel('Thresholds')
#     plt.title('TPR against Thresholds')
#     plt.legend(loc='best')
    
# def fit_n_plot(arg_train,arg_test,lab):
#     lr = linear_model.LogisticRegression(C=1e30)
#     lr.fit(arg_train.drop(lab,1), arg_train[lab])
#     mm = svm.SVC(kernel='linear')
#     mm.fit(arg_train.drop(lab,1), 2*arg_train[lab]-1)
#     plotAUC(arg_test[lab], lr.predict_proba(arg_test.drop(lab,1))[:,1], 'LR')    
#     plotAUC(arg_test[lab], mm.decision_function(arg_test.drop(lab,1)), 'SVM')
#     plt.show()

#compute the aucs for each test folds
def xValSVM(dataset,label_name,k,cs):
    cv = KFold(n=dataset.shape[0], n_folds = k)
    aucs = {}
    for train_index,test_index in cv:
        train_f = dataset.iloc[train_index]
        validation_f = dataset.iloc[test_index] 
        for c in cs:
            svmachine = svm.SVC(kernel='linear',C=c)
            svmachine.fit(train_f.drop(label_name,1),train_f[label_name])
#             print validation_f[label_name]
            metric = roc_auc_score(validation_f[label_name], svmachine.decision_function(validation_f.drop(label_name,1)))
            
            if (aucs.has_key(c)):
                aucs[c].append(metric)
            else:
                aucs[c] = [metric]
    return aucs

# def norm(dataset):
#     normalized = (dataset - dataset.mean())/(dataset.std())
#     return normalized
             
        
def execXValidation(dataset,label_name):
    extracted_dataset = dataset.drop(['DER_deltaeta_jet_jetmv','DER_mass_jet_jetmv','DER_prodeta_jet_jetmv','DER_mass_MMCmv'],1)
#     print extracted_dataset
#     print extracted_dataset.shape[1]
#     normalized_dataset = norm(extracted_dataset.ix[:,range(extracted_dataset.shape[1]-1)])
#     normalized_dataset[label_name]=extracted_dataset[label_name]
#     print normalized_dataset
    f = open('results.txt','a')
    cs = [10**i for i in range(-9,2)]
    aucs = xValSVM(extracted_dataset, label_name, 10, cs)  
    aucs_df = pd.DataFrame(aucs)
    print >> f, 'aucs_df:\n',aucs_df
    aucs_mean = aucs_df.mean(axis=0)
    print >> f, 'aucs_mean:\n',aucs_mean
    aucs_stderr = aucs_df.sem()
    print >> f, 'stderr(AUC):\n',np.std(aucs_df)
    aucs_lower_bound = aucs_mean-aucs_stderr
    aucs_upper_bound = aucs_mean+aucs_stderr
    aucs_array = aucs_df.values
    print >> f, 'array:\n',aucs_array
    #aucs_ref = pd.Series(np.max(aucs_array-np.std(aucs_array)),index=cs)
    aucs_ref = pd.Series(0.7305,index=cs)
    print >> f, 'ref:\n',aucs_ref
#     aucs_mean.plot(logx=True,legend=True,label='Mean',title='X-Validation on different C values')
#     aucs_lower_bound.plot(logx=True,style='k+',legend=True,label='Mean-Stderr')
#     aucs_upper_bound.plot(logx=True,style='k--',legend=True,label='Mean+Stderr')
#     aucs_ref.plot(logx=True,style='r',legend=True,label='Reference Line')
#     plt.show()   
    f.close()
    
# def testAUCBoot(test, nruns, model, lab):
#     '''
#     Samples with replacement, runs multiple eval attempts
#     returns all bootstrapped results
#     '''
#     auc_res = []; oops = 0
#     for i in range(nruns):
#         test_samp = test.iloc[np.random.randint(0, len(test), size=len(test))]
#         try:
#             auc_res.append(roc_auc_score(test_samp[lab], model.predict_proba(test_samp.drop(lab,1))[:,1]))
#         except:
#             oops += 1
#     return auc_res

# def modBootstrapper(train,test,nruns,sampsize,lr=1,c=None):
#     auc_res = []
#     oops = 0
#     for i in range(nruns):
#         train_samp = train.iloc[np.random.randint(0,len(train),size=sampsize)]
#         if lr == 0: 
#             train_samp_svm = svm.SVC(C=c,probability=True)
#             train_samp_svm.fit(train_samp.drop('Y',1),train_samp['Y'])
#             auc_res.append(roc_auc_score(test['Y'],train_samp_svm.predict_proba(test.drop('Y',1))[:,1]))
#         elif lr == 1:
#             train_samp_lr = linear_model.LogisticRegression()
#             train_samp_lr.fit(train_samp.drop('Y',1),train_samp['Y'])
#             auc_res.append(roc_auc_score(test['Y'],train_samp_lr.predict_proba(test.drop('Y',1))[:,1]))
#     print 'auc_res:\n' , auc_res
#     return np.mean(auc_res),np.std(auc_res)
# 
# def execBootstrap(train,test,nruns,sampsize):
#     #normalizing data
#     extracted_train = train.drop(['DER_deltaeta_jet_jetmv','DER_mass_jet_jetmv','DER_prodeta_jet_jetmv','DER_mass_MMCmv'],1)
#     extracted_test = test.drop(['DER_deltaeta_jet_jetmv','DER_mass_jet_jetmv','DER_prodeta_jet_jetmv','DER_mass_MMCmv'],1)
#     normalized_train = norm(extracted_train.ix[:,range(extracted_train.shape[1]-1)])
#     normalized_train['Y']=extracted_train['Y']
#     normalized_test = norm(extracted_test.ix[:,range(extracted_test.shape[1]-1)])
#     normalized_test['Y']=extracted_test['Y']
#     svm_aucs = []
#     lr_aucs = []
#     
#     for k in sampsize:
#         mean_auc_svm, std_auc_svm = modBootstrapper(normalized_train,normalized_test,nruns,k,lr=0,c=10)
#         mean_auc_lr, std_auc_lr = modBootstrapper(normalized_train, normalized_test, nruns, k, lr=1)
#         
#         print 'mean_svm:', mean_auc_svm, 'std_svm:',std_auc_svm,'\n'
#         svm_aucs.append([mean_auc_svm,std_auc_svm])
#         print 'mean_lr', mean_auc_lr, 'std_lr', std_auc_lr, '\n'
#         lr_aucs.append([mean_auc_lr,std_auc_lr])
#     
#     svm_df = pd.DataFrame(svm_aucs,index=sampsize)
#     lr_df = pd.DataFrame(lr_aucs,index=sampsize)
#     #lower bound and upper bound
#     svm_ub_df = svm_df[0]+svm_df[1]
#     svm_lb_df = svm_df[0]-svm_df[1]
#     lr_ub_df = lr_df[0]+lr_df[1]
#     lr_lb_df = lr_df[0]-lr_df[1] 
#     
#     svm_df[0].plot(logx = True,style='g',title='Bootstrap AUC Results',label='SVM Mean',legend=True)
#     lr_df[0].plot(logx = True,style='r',label='LR Mean',legend=True)
#     
#     lr_ub_df.plot(logx=True,style='r--',label='LR mean+stderr',legend=True)
#     lr_lb_df.plot(logx=True,style='r+',label='LR mean-stderr',legend=True)
#     
#     svm_ub_df.plot(logx=True,style='g--',label='SVM mean+stderr',legend=True)
#     svm_lb_df.plot(logx=True,style='g+',label='SVM mean-stderr',legend=True)
#     plt.show()
#     
#     
#         
#     
#         
#         
    
    
        

def main():
    train = cleanBosonData('boson_training_cut_2000.csv').drop('EventId',1)
    #print train
    test = cleanBosonData('boson_testing_cut.csv').drop('EventId',1)
#     fit_n_plot(train,test,'Y')
    execXValidation(train, 'Y')
#     execBootstrap(train, test, 20, [50, 100, 200, 500, 1000, 1500, 2000, 2500, 3000])
    
if __name__ == "__main__":
    main()