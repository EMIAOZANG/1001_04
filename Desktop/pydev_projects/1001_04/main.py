import pandas as pd
import numpy as np
import sys
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm, linear_model
from sklearn.neighbors import KNeighborsClassifier
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


def main():
    train = cleanBosonData('boson_training_cut_2000.csv').drop('EventID',axis=1)
    test = cleanBosonData('boson_training_cut.csv').drop('EventID',axis=1)
    fit_n_plot(train,test,'Y')
    
if __name__ == "__main__":
    main()