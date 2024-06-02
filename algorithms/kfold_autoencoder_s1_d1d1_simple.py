import keras
import pickle
from keras import layers
import pandas as pd
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Model, Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import tensorflow as tf
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


def find_threshold(model, scaled_train_data_x):
    reconstructions = model.predict(scaled_train_data_x)
    # provides losses of individual instances
    reconstruction_errors = tf.keras.losses.mse(reconstructions, scaled_train_data_x)
    # threshold for anomaly scores
    threshold = np.mean(reconstruction_errors.numpy()) \
                + np.std(reconstruction_errors.numpy())
    #threshold = np.percentile(reconstruction_errors, 80)
    print(threshold)
    # threshold = 0.1
    # plt.hist(reconstruction_errors[None, :], bins=20)
    # plt.xlabel("Train loss")
    # plt.ylabel("No of examples")
    # plt.show()

    return threshold


def get_predictions(model, scaled_test_data_x, threshold):
    predictions = model.predict(scaled_test_data_x)
    # provides losses of individual instances
    errors = tf.keras.losses.mse(predictions, scaled_test_data_x)

    # plt.hist(errors[None, :], bins=20)
    # plt.xlabel("Test loss")
    # plt.ylabel("No of examples")
    # plt.show()
    # 0 = anomaly, 1 = normal
    # anomaly_mask = pd.Series(errors) > threshold
    # preds = anomaly_mask.map(lambda x: 0.0 if x == True else 1.0)
    print("ERRORS SHAPE:", errors.shape)
    for i in range(len(errors)):
        if errors[i] < threshold:
            predicted_list.append(True)
        else:
            predicted_list.append(False)
    return tf.math.less(errors, threshold)
    # return predictions

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

test_data_x = df_test_all.iloc[:, :-1]
test_data_y = df_test_all.iloc[:, -1:]

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
class AutoEncoder(Model):
    def __init__(self):
        super().__init__()
        print("IN ENC")
        self.encoder = Sequential([
            keras.Input(shape=(556), name='enc'),
            # encoder_input = keras.Input(shape=(1500, 555), name ='enc')
            # encoder_flatten = keras.layers.Flatten()(encoder_input)
            # encoder_layer1 = keras.layers.Dense(36630, activation='sigmoid')(encoder_flatten)
            keras.layers.Dense(240, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(2, activation='softmax')
        ])
        self.decoder = Sequential([
            keras.layers.Dense(2, activation='softmax'),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(240, activation='relu'),
            # decoder_layer2 = keras.layers.Dense(36630, activation='sigmoid')(decoder_layer1)
            keras.layers.Dense(556, activation='sigmoid')
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded


model = AutoEncoder()
# model.compile(loss='mse', metrics=['mse'], optimizer='RMSprop')
model.compile(loss='mse', metrics=['mse'], optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.05))

predicted_list = []
fprs, tprs, scores = [], [], []

cv = StratifiedKFold(n_splits=4)

X = np.concatenate((scaled_train_data_x, scaled_test_data_x), axis=0)
y = np.concatenate((train_data_y.values.ravel(), test_data_y.values.ravel()), axis=0)

X = pd.DataFrame(X)
y = pd.DataFrame(y)

print(X.shape)
print(y.shape)
c_0 = 0
c_1 = 0
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
    history = model.fit(
        X.iloc[train],
        X.iloc[train],
        epochs=20,
        batch_size=2048,
        # batch_size=512
        validation_split=0.2
        #validation_data = (X.iloc[test], X.iloc[test])
    )
    #abc.fit(X.iloc[train], y.iloc[train].values.ravel())

    threshold = find_threshold(model, scaled_train_data_x)
    print(f"Threshold: {threshold}")
    # Threshold: 0.01001314025746261
    reconstructions = model.predict(scaled_train_data_x)
    train_loss = tf.keras.losses.mse(reconstructions, scaled_train_data_x)

    plt.hist(train_loss[None, :], bins=50)
    plt.xticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    plt.xlabel("Train loss")
    plt.ylabel("No of examples")
    plt.show()

    reconstructions = model.predict(scaled_test_data_x)
    test_loss = tf.keras.losses.mae(reconstructions, scaled_test_data_x)

    plt.hist(test_loss[None, :], bins=50)
    plt.xticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    plt.xlabel("Test loss")
    plt.ylabel("No of examples")
    plt.show()

    y_pred = get_predictions(model, scaled_test_data_x, threshold)
    test_data_y = y.iloc[test].values.ravel()
    # y_pred = abc.predict(scaled_test_data_x)
    accuracy = metrics.accuracy_score(y_pred, test_data_y)
    print(accuracy)
    print(confusion_matrix(test_data_y, y_pred))
    print(classification_report(test_data_y, y_pred))
    print(accuracy_score(test_data_y, y_pred))

    precision = precision_score(test_data_y, y_pred)
    print('Precision: %.3f', precision)
    recall = recall_score(test_data_y, y_pred)
    print('Recall: %.3f', recall)
    score = f1_score(test_data_y, y_pred)
    print('F-Measure: %.3f', score)

    # _, _, auc_score_train = compute_roc_auc(train)
    # fpr, tpr, auc_score = compute_roc_auc(test)

    fpr, tpr, thresholds = roc_curve(test_data_y, y_pred)
    roc_auc = auc(fpr, tpr)
    #scores.append((auc_score_train, auc_score))
    fprs.append(fpr)
    tprs.append(tpr)


    predicted_list = []
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
    plt.title('ROC curve for AutoEncoder')
    plt.legend(loc='best')
    # plt.savefig('RF_no_shuffle_smooth2.png')
    plt.savefig('AE_scenario_s1_d1d1.png')
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
    plt.title('ROC curve for AE')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig('AE_scenario_s1_d3d3.png',  bbox_inches='tight')
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




