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
import matplotlib.pyplot as plt

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

def plotAUC(truth, pred, lab):
    fpr, tpr, thresholds = roc_curve(truth, pred)
    roc_auc = auc(fpr, tpr)
    c = (np.random.rand(), np.random.rand(), np.random.rand())
    plt.plot(fpr, tpr, color=c, label= lab+' (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC')
    plt.legend(loc="lower right")
    
def fit_n_plot(arg_train,arg_test,lab):
    lr = linear_model.LogisticRegression(C=1e30)
    lr.fit(arg_train.drop(lab,1), arg_train[lab])
    mm = svm.SVC(kernel='linear')
    mm.fit(arg_train.drop(lab,1), 2*arg_train[lab]-1)
    plotAUC(arg_test[lab], lr.predict_proba(arg_test.drop(lab,1))[:,1], 'LR')    
    plotAUC(arg_test[lab], mm.decision_function(arg_test.drop(lab,1)), 'SVM')
    plt.show()

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

def norm(dataset):
    normalized = (dataset - dataset.mean())/(dataset.max()-dataset.min())
    return normalized
         
        
def execXValidation(dataset,label_name):
    extracted_dataset = dataset.drop(['DER_deltaeta_jet_jetmv','DER_mass_jet_jetmv','DER_prodeta_jet_jetmv','DER_mass_MMCmv'],1)
#     print extracted_dataset
#     print extracted_dataset.shape[1]
    normalized_dataset = norm(extracted_dataset.ix[:,range(extracted_dataset.shape[1]-1)])
    normalized_dataset[label_name]=extracted_dataset[label_name]
#     print normalized_dataset
    cs = [10**i for i in range(-9,2)]
    aucs = xValSVM(normalized_dataset, label_name, 10, cs)  
    aucs_df = pd.DataFrame(aucs)
    print aucs_df
    aucs_mean = aucs_df.mean(axis=0)
    print aucs_mean
    aucs_stderr = aucs_df.sem()
    aucs_lower_bound = aucs_mean-aucs_stderr
    aucs_upper_bound = aucs_mean+aucs_stderr
    aucs_ref = np.max(aucs_df-aucs_stderr)
    print aucs_ref
    aucs_mean.plot(logx=True)
    aucs_lower_bound.plot(logx=True,style='k+')
    aucs_upper_bound.plot(logx=True,style='k--')
    aucs_ref.plot(logx=True,style='r')
    plt.show()   

#def modBootsrapper(train,test,nruns,sampsize,lr=1,c=None):
    
    

def main():
    train = cleanBosonData('boson_training_cut_2000.csv').drop('EventId',1)
    print train
    test = cleanBosonData('boson_testing_cut.csv').drop('EventId',1)
    #fit_n_plot(train,test,'Y')
    execXValidation(train, 'Y')
    
if __name__ == "__main__":
    main()