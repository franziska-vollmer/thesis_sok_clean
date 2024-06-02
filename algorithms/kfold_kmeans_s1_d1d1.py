import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn import metrics
#from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

file1 = open('apps-sok.txt', 'r')


#train_path = 'train_test_supervised_with_timestamp/'
#test_path  = 'train_test_supervised_with_timestamp/'

train_path = 'train_autoencoder_with_timestamp/'
test_path  = 'test_autoencoder_with_timestamp/'
#train_path = 'train_test_supervised_with_timestamp/'
#test_path  = 'train_test_supervised_with_timestamp/'

file1 = open('apps-sok-first-part.txt', 'r')
Lines1= file1.readlines()

file2 = open('apps-sok-first-part.txt', 'r')
Lines2= file2.readlines()
list_of_train = [1,2,3,4]
list_of_test =  [1,2,3,4]


#Lines2 = file2.readlines()
# list_of_train = [1, 2, 3, 4]
# list_of_test = [1, 2, 3, 4]
df_train_all = pd.DataFrame()
df_test_all = pd.DataFrame()
predicted_list = []


def compute_roc_auc_train(scaled_train_data_x, train_data_y):
    y_predict = rfc.predict(scaled_train_data_x)
    fpr, tpr, thresholds = roc_curve(train_data_y, y_predict)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score

def compute_roc_auc_test(scaled_test_data_x, test_data_y):
    y_predict = rfc.predict(scaled_test_data_x)
    fpr, tpr, thresholds = roc_curve(test_data_y, y_predict)
    print("FPR:", fpr)
    print("TPR:", tpr)
    print("THR:", thresholds)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score

def compute_roc_auc(index):
    y_predict = abc.predict_proba(X.iloc[index])[:, 1]
    fpr, tpr, thresholds = roc_curve(y.iloc[index], y_predict)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score


def display(results):
    print(f'Best parameters are: {results.best_params_}')
    print("\n")
    mean_score = results.cv_results_['mean_test_score']
    std_score = results.cv_results_['std_test_score']
    params = results.cv_results_['params']
    for mean, std, params in zip(mean_score, std_score, params):
        print(f'{round(mean, 3)} + or -{round(std, 3)} for the {params}')


count = 0
for line in Lines1:

    content = line.strip()
    # print(content)
    for k in list_of_train:
        path = train_path + content + '-' + str(k) + '.pkl'
        # path = 'C:/Users/sadun/PycharmProjects/cdl_remade/venv/data-CDL/' + content + '/' + content + '-' + str(k) + '_freqvector_full.csv'
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
    # print(content)
    for ki in list_of_test:
        path = test_path + content + '-' + str(ki) + '.pkl'
        # path = 'C:/Users/sadun/PycharmProjects/cdl_remade/venv/data-CDL/' + content + '/' + content + '-' + str(k) + '_freqvector_full.csv'
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

# exit()
# df_test_reversed = df_test_all[::-1]
# df_test_all.drop(df_test_all.index, inplace=True)
# df_test_all = df_test_reversed
print(df_train_all.shape)
print(df_test_all.shape)

# With time
# train_data_x = df_train_all.iloc[:, :-1]
# train_data_y = df_train_all.iloc[:, -1:]
#
# test_data_x = df_test_all.iloc[:, :-1]
# test_data_y = df_test_all.iloc[:, -1:]


train_data_x = df_train_all.iloc[:, :-1]
train_data_y = df_train_all.iloc[:, -1:]


list_train_data_y = train_data_y[train_data_y.columns[0]].tolist()

    #print(list_train_data_y)
    #exit()

test_data_x = df_test_all.iloc[:, :-1]
test_data_y = df_test_all.iloc[:, -1:]

    #print(type(test_data_y))

list_test_data_y = test_data_y[test_data_y.columns[0]].tolist()

# Example
# train_data_x = df_train_all.iloc[:4155, :-1]
# train_data_y = df_train_all.iloc[:4155, -1:]

# test_data_x = df_test_all.iloc[:4155, :-1]
# test_data_y = df_test_all.iloc[:4155, -1:]


print(train_data_x.shape)
print(test_data_x.shape)

sc = StandardScaler()
# sc = Normalizer()
scaled_train_data_x = sc.fit_transform(train_data_x)
scaled_test_data_x = sc.fit_transform(test_data_x)
print(type(scaled_train_data_x))
# exit()

# train_data_y = sc.fit_transform(train_data_y)
# test_data_y  = sc.fit_transform(test_data_y)


# print(train_data_y.values.ravel())
# rfc = RandomForestClassifier(n_estimators=200, max_depth=16, max_features=100)
# rfc = RandomForestClassifier(n_estimators=200, class_weight='balanced')

# dict_weights = {0:16.10, 1:0.51}

#abc = AdaBoostClassifier(n_estimators=200)

rfc = KMeans(n_clusters=10)
threshold=85

predicted_list = []
fprs, tprs, scores = [], [], []

cv = StratifiedKFold(n_splits=4, shuffle=True)

X = np.concatenate((scaled_train_data_x, scaled_test_data_x), axis=0)
y = np.concatenate((train_data_y.values.ravel(), test_data_y.values.ravel()), axis=0)

X = pd.DataFrame(X)
y = pd.DataFrame(y)

print(X.shape)
print(y.shape)
c_0 = 0
c_1 = 0
k_count=1
for (train, test), i in zip(cv.split(X, y), range(4)):
    c_0 = 0
    c_1 = 0
    #print("Anomaly", y.iloc[test][0] == 0)
    #print("Normal", y.iloc[train][0] == 1)
    #print(type(y[train][0]))
    y_list = y.iloc[train].values.tolist()
    #print(type(y_list))
    #print(y_list)
    scaled_train_data_x = X.iloc[train]
    scaled_test_data_x = X.iloc[test]
    cv_test_data_y = y.iloc[test].values.ravel()
    cv_train_data_y = y.iloc[train].values.ravel()
    print(scaled_train_data_x.shape, cv_train_data_y.shape)
    print(scaled_test_data_x.shape, cv_test_data_y.shape)
    #exit()
    for i in range(len(y_list)):
        if y_list[i][0] == 0:
            c_0 += 1
            #print(i)
        if y_list[i][0] == 1:
            c_1 += 1
            #print(i)
    print("Anomaly:", c_0)
    print("Normal:", c_1)
    #exit()

    rfc.fit(scaled_train_data_x)
    centre = rfc.cluster_centers_
    print("CENTRE:", rfc.cluster_centers_)
    # print(type(centre))
    # exit()

    # for centre_iterator in range(len(centre)):
    #     print("ITE: ",centre_iterator, centre[centre_iterator])

    # exit()
    # print("LABELS:", rfc.labels_)
    # print(set(rfc.labels_))

    _, _, auc_score_train = compute_roc_auc_train(scaled_train_data_x, cv_train_data_y)
    fpr, tpr, auc_score = compute_roc_auc_test(scaled_test_data_x, cv_test_data_y)
    scores.append((auc_score_train, auc_score))
    fprs.append(fpr)
    tprs.append(tpr)
    # scaled_test_data_x = X.iloc[test]
    # test_data_y = y.iloc[test].values.ravel()

    #y_pred = rfc.predict(scaled_test_data_x)
    #With train data



    #exit()
    #With test data
    print("With test data")
    y_pred = rfc.predict(scaled_test_data_x)

    y_pred_centres = rfc.cluster_centers_
    print("C", y_pred_centres.shape)
    print("CC", y_pred_centres[y_pred].shape)
    # print(scaled_train_data_x)
    # exit()

    a = np.array([1, 2, 3, 4])
    b = np.array([7, 8])

    # print(a[b])
    # exit()

    # print(y_pred_centres)
    # print(y_pred_centres[y_pred])
    # exit()

    # dist = [np.linalg.norm(x - y) for x, y in zip(scaled_train_data_x[cv_train_data_y], y_pred_centres[y_pred])]
    # print(y_pred_centres[0])
    dist = []
    min_20 = []
    sum = 0
    # print(len(y_pred_centres))
    # print(scaled_train_data_x)
    # print(len(range(scaled_train_data_x)))
    # exit()

    # scaled_train_data_x_np = scaled_train_data_x.to_numpy()
    # y_pred_centres_np = y_pred_centres
    # for iterator1 in range(len(scaled_train_data_x_np)):
    #     for iterator2 in range(len(y_pred_centres_np)):
    #         distance = [np.linalg.norm(x - y) for x, y in zip(scaled_train_data_x_np[iterator1], y_pred_centres_np[iterator2])]
    #         #dist = [np.linalg.norm(scaled_train_data_x[0] - y_pred_centres[0])]
    #         #print(len(dist))
    #         #print(dist)
    #         for iterator3 in range(len(distance)):
    #             #print(distance[iterator3])
    #             sum = sum + (distance[iterator3] * distance[iterator3])
    #
    #
    #         #print(sum)
    #         distance = sum**(1/556)
    #         #print(distance)
    #         #exit()
    #         min_20.append(distance)
    #         #print(iterator1, iterator2)
    #     #print(iterator1)
    #     #exit()
    #     minimum_centroid_distance = min(min_20)
    #     dist.append(minimum_centroid_distance)
    #     min_20 = []
    #     sum = 0

    # print(dist)
    # exit()

    #dist=[]
    scaled_test_data_x_np = scaled_test_data_x.to_numpy()
    dist2 = [np.linalg.norm(x - y) for x, y in zip(scaled_test_data_x_np, y_pred_centres[y_pred])]
    dist2_y_pred = np.array(dist2)
    #print(dist2_y_pred.shape)
    # exit()
    # print()
    # print("HERE")
    # print(dist2)
    # print(len(dist2))
    print(dist2_y_pred)
    # plt.hist(dist2_y_pred[None, :], bins=50)
    # #plt.xticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    # plt.xlabel("Test loss")
    # plt.ylabel("No of examples")
    # plt.show()
    #exit()
    #threshold = 50

    #threshold = np.percentile(dist2, threshold)
    threshold = np.mean(dist2_y_pred) + np.std(dist2_y_pred)
    print("MEAN", np.mean(dist2_y_pred))

    print("THRESH", threshold)
    print(np.percentile(dist2, 20))
    print(np.percentile(dist2, 40))
    print(np.percentile(dist2, 50))
    print(np.percentile(dist2, 80))
    print(np.percentile(dist2, 90))

    list_of_predictors2 = []
    # print((rfc.labels_.shape))
    # c0 = 0
    # c1 = 0
    # for x in (rfc.labels_):
    #     if x == 0:
    #         c0 += 1
    #     elif x == 1:
    #         c1 += 1
    #
    # print("CO", c0)
    # print("C1", c1)
    # exit()

    # dist_y_pred[dist >= np.percentile(dist, threshold)] = 1
    # dist_y_pred[dist < np.percentile(dist, threshold)] = 0

    # print(type(cv_test_data_y))
    # print(cv_test_data_y.shape)
    # print(type(dist2_y_pred))
    # train_data_y_np = cv_train_data_y.to_numpy()
    # print(type(train_data_y_np))
    actual_anomaly = np.where(cv_test_data_y == 0)[0]
    actual_benign = np.where(cv_test_data_y == 1)[0]
    anomalies_indices = np.where(dist2_y_pred > threshold)[0]
    normal_indices = np.where(dist2_y_pred <= threshold)[0]

    # dist_y_pred[dist >= np.percentile(dist, threshold)] = 0
    # dist_y_pred[dist < np.percentile(dist, threshold)] = 1

    begin_iteration = 0
    list_of_predictors2=[]
    x = 0
    for begin_iteration in dist2_y_pred:
        #print(begin_iteration, threshold)
        if begin_iteration > threshold:
            #print(x)
            list_of_predictors2.append(0)
        else:
            list_of_predictors2.append(1)
            # print(x)
        x += 1

    #exit()
    list_of_predictors2 = np.array(list_of_predictors2)
    #print(list_of_predictors2[1364:1372])
    #print(cv_train_data_y[1364:1372])
    list_of_predictors2 = list_of_predictors2.tolist()

    print("MEAN ANOMALY ACTUAL", np.mean(dist2_y_pred[actual_anomaly]))

    #print("Anomaly Dist", dist2_y_pred[actual_anomaly])
    #exit()

    # Print the results
    # print("Anomaly Normal:", actual_anomaly)
    # print("Benign Normal:", actual_benign)
    # print("Anomaly Indices:", anomalies_indices)
    # print("Benign Indices:", normal_indices)

    #exit()

    print("predicted {} count {}".format(np.sum(dist2_y_pred), len(dist2_y_pred)))

    list_of_acuracy = []
    list_of_precision = []
    list_of_recall = []
    list_of_f1 = []

    # list_of_predictors = y_pred
    # list_of_thresholds = [1, 10, 20,30,40,50,60,70,80,90,99]
    #print(cv_train_data_y.shape, len(list_of_predictors), dist_y_pred.shape)

    #cv_test_data_y = cv_test_data_y.tolist()
    #exit()
    accuracy = metrics.accuracy_score(cv_test_data_y, list_of_predictors2)
    print(accuracy)
    print(confusion_matrix(cv_test_data_y, list_of_predictors2))
    print(classification_report(cv_test_data_y, list_of_predictors2))
    print(accuracy_score(cv_test_data_y, list_of_predictors2))

    precision = precision_score(cv_test_data_y, list_of_predictors2, average='macro')
    print('Precision: %.3f', precision)
    recall = recall_score(cv_test_data_y, list_of_predictors2, average='macro')
    print('Recall: %.3f', recall)
    score = f1_score(cv_test_data_y, list_of_predictors2, average='macro')
    print('F-Measure: %.3f', score)
    #exit()
    #break


def plot_roc_curve_simple(fprs, tprs):
    plt.figure(figsize=(8, 8))
    for i in range(len(fprs)):
        roc_auc = auc(fprs[i], tprs[i])
        ysmoothed = gaussian_filter1d(tprs[i], sigma=2)
        plt.plot(fprs[i], tprs[i], label='ROC fold %d (AUC = %0.2f)' % (i + 1, roc_auc))
        i += 1
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for K-Means')
    plt.legend(loc='best')
    # plt.savefig('RF_no_shuffle_smooth2.png')
    plt.savefig('KM_scenario_s1_d2d2.png')
    plt.show()


plot_roc_curve_simple(fprs, tprs)
pd.DataFrame(scores, columns=['AUC Train', 'AUC Test'])




exit()



    # opt = tf.keras.RMSprop(0.001, decay=1e-6)
    # autoencoder = keras.Model(encoder_input, decoder_output, name='autoencoder')
    # autoencoder.summary()




threshold = find_threshold(model, scaled_train_data_x)
print(f"Threshold: {threshold}")
# Threshold: 0.01001314025746261
predictions = get_predictions(model, scaled_test_data_x, threshold)
print(predictions.shape)
print(scaled_test_data_x.shape)
print(predictions)
print(scaled_test_data_x)
# y_pred=predictions
# y_test=dataset_test
# y_pred=np.argmax(predictions, axis=1)
# y_test=np.argmax(dataset_test, axis=1)
# print(y_pred)
# print(y_test)
# cm = confusion_matrix(y_test, y_pred)
# print(cm)

fpr, tpr, thresholds = roc_curve(test_data_y, predictions)
auc_score = auc(fpr, tpr)
fprs.append(fpr)
tprs.append(tpr)

print(confusion_matrix(test_data_y, predictions))
print(classification_report(test_data_y, predictions))

print("Accuracy :", accuracy_score(test_data_y, predictions))
# print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(test_data_y, predictions))
print("Recall   :", recall_score(test_data_y, predictions))
# Epoch 1, threshold 0.00287, accuracy 92.247
score = f1_score(test_data_y, predictions)
print('F-Measure: %.3f', score)

print(tf.get_static_value(predictions[5]))
print(type(tf.get_static_value(predictions[5])))
if (tf.get_static_value(predictions[5]) == 'True'):
    print("HAPPY")

total_number = 0
true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0
number_of_labels = len(test_data_y)
test_labels = test_data_y
lines_of_fpr = []
lines_of_fnr = []




def plot_roc_curve_simple(fprs, tprs):
    plt.figure(figsize=(8,8))
    for i in range(len(fprs)):
        roc_auc = auc(fprs[i], tprs[i])
        ysmoothed = gaussian_filter1d(tprs[i], sigma=2)
        plt.plot(fprs[i], tprs[i], label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc))
        i +=1
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for KM')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig('KM_scenario_s1_d1d1.png',  bbox_inches='tight')
    plt.show()




plot_roc_curve_simple(fprs, tprs)

exit()


for i in range(len(test_labels)):
    if test_labels[i] == 1 and predicted_list[i] is True:
        true_positive +=1
        #print("TP")
    if test_labels[i] == 0 and predicted_list[i] is False:
        true_negative +=1
    if test_labels[i] == 0 and predicted_list[i] is True:
        false_negative +=1
        lines_of_fnr.append(i)
    if test_labels[i] == 1 and predicted_list[i] is False:
        false_positive +=1
        lines_of_fpr.append(i)

tpr = (true_positive/i)*100
tnr = (true_negative/i)*100
fpr = (false_positive/i)*100
fnr = (false_negative/i)*100

print("FALSE NEGATIVE:", false_negative)
print("FALSE POSITIVE:", false_positive)
print("TPR:", tpr)
print("TNR:", tnr)
print("FPR:", fpr)
print("FNR:", fnr)
#print(lines_of_fpr)
#print(lines_of_fnr)




