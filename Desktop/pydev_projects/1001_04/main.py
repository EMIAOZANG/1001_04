import pandas as pd
import numpy as np
import sys

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

def main():
    cleanBosonData('boson_training_cut_2000.csv')
    
if __name__ == "__main__":
    main()