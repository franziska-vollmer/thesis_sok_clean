import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

#%matplotlib inline
matplotlib.rcParams.update({'font.size': 20})


supervised_path = 'train_test_supervised_with_timestamp/'
file1 = open('apps-sok-first-part.txt', 'r')
Lines = file1.readlines()
file2 = open('apps-sok-first-part.txt', 'r')
Lines2 = file2.readlines()
list_of_train = [1,2,3,4]
list_of_test  = [1,2,3,4]
df_train_all = pd.DataFrame()
df_test_all  = pd.DataFrame()


def plot_roc_curve(fprs, tprs):
    """Plot the Receiver Operating Characteristic from a list
    of true positive rates and false positive rates."""
    
    # Initialize useful lists + the plot axes.
    tprs_interp = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    f, ax = plt.subplots(figsize=(14,10))
    
    # Plot ROC for each K-Fold + compute AUC scores.
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        #tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        #tprs_interp[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        
    # Plot the luck line.
    # plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
    #          label='Luck', alpha=.8)
    
    # Plot the mean ROC.
    # mean_tpr = np.mean(tprs_interp, axis=0)
    # mean_tpr[-1] = 1.0
    # mean_auc = auc(mean_fpr, mean_tpr)
    # std_auc = np.std(aucs)
    # ax.plot(mean_fpr, mean_tpr, color='b',
    #          label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
    #          lw=2, alpha=.8)
    #
    # # Plot the standard deviation around the mean ROC.
    # std_tpr = np.std(tprs_interp, axis=0)
    # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
    #                  label=r'$\pm$ 1 std. dev.')
    
    # Fine tune and show the plot.
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")
    plt.show()
    return (f, ax)

def compute_roc_auc(index):
    y_predict = rfc.predict_proba(X.iloc[index])[:,1]
    fpr, tpr, thresholds = roc_curve(y.iloc[index], y_predict)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score


def display(results):
    print(f'Best parameters are: {results.best_params_}')
    print("\n")
    mean_score = results.cv_results_['mean_test_score']
    std_score = results.cv_results_['std_test_score']
    params = results.cv_results_['params']
    for mean,std,params in zip(mean_score,std_score,params):
        print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')


count = 0
for line in Lines:

    content = line.strip()
    #print(content)
    for k in list_of_train:
        path = supervised_path  + content + '-' + str(k) +'.pkl'
        #path = 'C:/Users/sadun/PycharmProjects/cdl_remade/venv/data-CDL/' + content + '/' + content + '-' + str(k) + '_freqvector_full.csv'
        print(path)
        picklefile_train = open(path, 'rb')
        df_individual_train = pickle.load(picklefile_train)
        has_nan = df_individual_train.isnull().any().any()
        if has_nan == True:
            print(path)
            exit()
        picklefile_train.close()
        df_train_all = pd.concat([df_train_all, df_individual_train], axis=0)
        #break
    #break



for line in Lines2:

    content = line.strip()
    #print(content)
    for ki in list_of_test:
        path = supervised_path + content + '-' + str(ki) +'.pkl'
        #path = 'C:/Users/sadun/PycharmProjects/cdl_remade/venv/data-CDL/' + content + '/' + content + '-' + str(k) + '_freqvector_full.csv'
        print(path)
        picklefile_test = open(path, 'rb')
        df_individual_test = pickle.load(picklefile_test)
        has_nan = df_individual_test.isnull().any().any()
        if has_nan == True:
            print(path)
            exit()
        picklefile_test.close()
        df_test_all = pd.concat([df_test_all, df_individual_test], axis=0)

        #break
    #break

#exit()
#df_test_reversed = df_test_all[::-1]
#df_test_all.drop(df_test_all.index, inplace=True)
#df_test_all = df_test_reversed
print(df_train_all.shape)
print(df_test_all.shape)

#With time
# train_data_x = df_train_all.iloc[:, :-1]
# train_data_y = df_train_all.iloc[:, -1:]
#
# test_data_x = df_test_all.iloc[:, :-1]
# test_data_y = df_test_all.iloc[:, -1:]




train_data_x = df_train_all.iloc[:, :-1]
train_data_y = df_train_all.iloc[:, -1:]

test_data_x = df_test_all.iloc[:, :-1]
test_data_y = df_test_all.iloc[:, -1:]


#Example
#train_data_x = df_train_all.iloc[:4155, :-1]
#train_data_y = df_train_all.iloc[:4155, -1:]

#test_data_x = df_test_all.iloc[:4155, :-1]
#test_data_y = df_test_all.iloc[:4155, -1:]


print(train_data_x.shape)
print(test_data_x.shape)

sc = StandardScaler()
#sc = Normalizer()
scaled_train_data_x = sc.fit_transform(train_data_x)
scaled_test_data_x = sc.fit_transform(test_data_x)
print(type(scaled_train_data_x))
#exit()

#train_data_y = sc.fit_transform(train_data_y)
#test_data_y  = sc.fit_transform(test_data_y)


#print(train_data_y.values.ravel())
#rfc = RandomForestClassifier(n_estimators=200, max_depth=16, max_features=100)
#rfc = RandomForestClassifier(n_estimators=200, class_weight='balanced')

#dict_weights = {0:16.10, 1:0.51}


rfc = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None,
        n_jobs=-1, n_neighbors=5, p=2, weights='uniform')

parameters = {
    "n_estimators":[200, 200, 200, 200, 200],
    "max_features":[100, 200, 300, 400, 500]
}


fprs, tprs, scores = [], [], []

cv = StratifiedKFold(n_splits=4, random_state=42, shuffle=True)

X = np.concatenate((scaled_train_data_x, scaled_test_data_x), axis=0)
y = np.concatenate((train_data_y.values.ravel(), test_data_y.values.ravel()), axis=0)

X=pd.DataFrame(X)
y=pd.DataFrame(y)

print(X.shape)
print(y.shape)

for (train, test), i in zip(cv.split(X, y), range(4)):
    rfc.fit(X.iloc[train], y.iloc[train].values.ravel())
    _, _, auc_score_train = compute_roc_auc(train)
    fpr, tpr, auc_score = compute_roc_auc(test)
    scores.append((auc_score_train, auc_score))
    fprs.append(fpr)
    tprs.append(tpr)
    scaled_test_data_x = X.iloc[test]
    test_data_y = y.iloc[test].values.ravel()
    y_pred = rfc.predict(scaled_test_data_x)
    accuracy = metrics.accuracy_score(y_pred, test_data_y)
    print(accuracy)
    print(confusion_matrix(test_data_y, y_pred))
    print(classification_report(test_data_y, y_pred))
    print(accuracy_score(test_data_y, y_pred))

    precision = precision_score(test_data_y, y_pred, average='macro')
    print('Precision: %.3f', precision)
    recall = recall_score(test_data_y, y_pred, average='macro')
    print('Recall: %.3f', recall)
    score = f1_score(test_data_y, y_pred, average='macro')
    print('F-Measure: %.3f', score)



def plot_roc_curve_simple(fprs, tprs):
    plt.figure(figsize=(8,8))  
    for i in range(len(fprs)):
        roc_auc = auc(fprs[i], tprs[i])
        ysmoothed = gaussian_filter1d(tprs[i], sigma=2)
        plt.plot(fprs[i], tprs[i], label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc))
        i +=1
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for KNN')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig('KNN_scenario_s1_d1d1.png',  bbox_inches='tight')
    plt.show()




plot_roc_curve_simple(fprs, tprs);
pd.DataFrame(scores, columns=['AUC Train', 'AUC Test'])




# #Below The 4fold
# #rfc.fit(scaled_train_data_x, train_data_y.values.ravel())
# #rfc.fit(scaled_train_data_x, train_data_y)
# y_pred = rfc.predict(scaled_test_data_x)
# print(y_pred)
# print(y_pred.shape)
# accuracy = metrics.accuracy_score(y_pred, test_data_y)
#
# #apple
# #---------------Metrics------------
#
# print(accuracy)
# print(confusion_matrix(test_data_y,y_pred))
# print(classification_report(test_data_y,y_pred))
# print(accuracy_score(test_data_y, y_pred))
#
#
# precision = precision_score(test_data_y, y_pred)
# print('Precision: %.3f', precision)
# recall = recall_score(test_data_y, y_pred)
# print('Recall: %.3f', recall)
# score = f1_score(test_data_y, y_pred)
# print('F-Measure: %.3f', score)
