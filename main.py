import math
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
import xgboost
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import numpy as np
from catboost import CatBoostRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
import joblib
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBClassifier

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

# Replace with data path
data_path = "C:/Users/IoT Lab/OneDrive - University of the West of Scotland/desktop/Health-spa Dataset/"

# Read data files
f1 = pd.read_csv(data_path + "Final_Dataframe_P1.csv")
f2 = pd.read_csv(data_path + "Final_Dataframe_P2.csv")
f3 = pd.read_csv(data_path + "Final_Dataframe_P3.csv")
f4 = pd.read_csv(data_path + "Final_Dataframe_P4.csv")
f5 = pd.read_csv(data_path + "Final_Dataframe_P5.csv")
f6 = pd.read_csv(data_path + "Final_Dataframe_P6.csv")
f7 = pd.read_csv(data_path + "Final_Dataframe_P7.csv")
f8 = pd.read_csv(data_path + "Final_Dataframe_P8.csv")
f9 = pd.read_csv(data_path + "Final_Dataframe_P9.csv")
f10 = pd.read_csv(data_path + "Final_Dataframe_P10.csv")
f11 = pd.read_csv(data_path + "Final_Dataframe_P11.csv")
f12 = pd.read_csv(data_path + "Final_Dataframe_P12.csv")
f13 = pd.read_csv(data_path + "Final_Dataframe_P13.csv")
f14 = pd.read_csv(data_path + "Final_Dataframe_P14.csv")

dataset = pd.concat([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14], ignore_index=True)
print("Dataset size: ", len(dataset), end="\n")

print("Dataset statistics: ", dataset.describe(), end="\n")

dataset.rename(columns={"Lux": "Light", "PM4": "Nr.particles0.3", "PM5": "Nr.particles0.5", "PM6": "Nr.particles1.0"}, inplace=True)

X = dataset[["Light","Nr.particles0.3","Nr.particles0.5","Nr.particles1.0","Temperature","Humidity","Pressure", "Sound", "IAQ"]]
y1 = dataset["Heart Rate"]
y2 = dataset["Skin Resistance"]
#Split dataset for train and test subsets
X1_train, X1_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.15)
X2_train, X2_test, y2_train, y2_test = train_test_split(X, y2, test_size=0.15)

# prepare the cross-validation procedure
#cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)

############# Training #################

####### Linear Regression
linparameters = {'fit_intercept':('True', 'False')}
rh = LinearRegression()
regheart = GridSearchCV(estimator=rh, param_grid=linparameters, cv=5, scoring='neg_mean_absolute_error')
regheart.fit(X1_train, y1_train)
print("Best param for linheart: ", regheart.best_params_, regheart.best_score_)
# evaluate model
#sscores = cross_val_score(regheart, X, y1, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
#ascores = cross_val_score(regheart, X, y1, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# report performance
#print('RMSE heart_lin: %.3f (%.3f)' % (np.mean(sscores), np.std(sscores)))
#print('MAE heart_lin: %.3f (%.3f)' % (np.mean(ascores), np.std(ascores)))


rs = LinearRegression()
regskin = GridSearchCV(estimator=rs, param_grid=linparameters, cv=5, scoring='neg_mean_absolute_error')
regskin.fit(X2_train, y2_train)
print("Best param for linskin: ", regskin.best_params_["fit_intercept"], regskin.best_score_)
# evaluate model
#sscores = cross_val_score(regskin, X, y2, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
#ascores = cross_val_score(regskin, X, y2, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# report performance
#print('RMSE skin_lin: %.3f (%.3f)' % (np.mean(sscores), np.std(sscores)))
#print('MAE skin_lin: %.3f (%.3f)' % (np.mean(ascores), np.std(ascores)))



####### Random Forest
forestparameters = {"criterion": ["squared_error", "poisson"], "max_depth": [None], "max_features": [6]}
fh = RandomForestRegressor()
regheart_rf = GridSearchCV(estimator=fh, param_grid=forestparameters, cv=5, scoring='neg_mean_absolute_error')
regheart_rf.fit(X1_train, y1_train)
print("Best param for rfheart: ", regheart_rf.best_params_, regheart_rf.best_score_)
#ascores = cross_val_score(regheart_rf, X, y1, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
#sscores = cross_val_score(regheart_rf, X, y1, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
# report performance
#print('RMSE heart_rf: %.3f (%.3f)' % (np.mean(sscores), np.std(sscores)))
#print('MAE heart_rf: %.3f (%.3f)' % (np.mean(ascores), np.std(ascores)))


fs = RandomForestRegressor()
regskin_rf = GridSearchCV(estimator=fs, param_grid=forestparameters, cv=5, scoring='neg_mean_absolute_error')
regskin_rf.fit(X2_train, y2_train)
print("Best param for rfskin: ", regskin_rf.best_params_, regskin_rf.best_score_)
#ascores = cross_val_score(regskin_rf, X, y2, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
#sscores = cross_val_score(regskin_rf, X, y2, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
# report performance
#print('RMSE skin_rf: %.3f (%.3f)' % (np.mean(ascores), np.std(ascores)))
#print('MAE skin_rf: %.3f (%.3f)' % (np.mean(sscores), np.std(sscores)))

####### XGB Regressor
xgbparameters = {"eta": [0.1, 0.15], "max_depth": [16, 25], "min_child_weight": [2, 3], "colsample_bytree": [0.67], "eval_metric": ["mae"]}
xgh = xgboost.XGBRegressor()
xgheart = GridSearchCV(estimator=xgh, param_grid=xgbparameters, cv=5, scoring='neg_mean_absolute_error')
xgheart.fit(X1_train, y1_train)
print("Best param for xgheart: ", xgheart.best_params_, xgheart.best_score_)

xgs = xgboost.XGBRegressor()
xgskin = GridSearchCV(estimator=xgs, param_grid=xgbparameters, cv=5, scoring='neg_mean_absolute_error')
xgskin.fit(X2_train, y2_train)
print("Best param for xgskin: ", xgskin.best_params_, xgskin.best_score_)

############# Testing #################
iteration = 0
rhmae = []
rhrmse = []
rsmae = []
rsrmse = []
fhmae = []
fhrmse = []
fsmae = []
fsrmse = []
xhmae = []
xhrmse = []
xsmae = []
xsrmse = []
fhmape = []
fsmape = []
rh = LinearRegression(fit_intercept=regheart.best_params_["fit_intercept"])
rs = LinearRegression(fit_intercept=regheart.best_params_["fit_intercept"])
fh = RandomForestRegressor(criterion=regheart_rf.best_params_["criterion"],
                           max_depth=regheart_rf.best_params_["max_depth"],
                           max_features=regheart_rf.best_params_["max_features"])
fs = RandomForestRegressor(criterion=regskin_rf.best_params_["criterion"],
                           max_depth=regskin_rf.best_params_["max_depth"],
                           max_features=regskin_rf.best_params_["max_features"])
xgh = xgboost.XGBRegressor(eta=xgheart.best_params_["eta"], max_depth=xgheart.best_params_["max_depth"],
                           min_child_weight=xgheart.best_params_["min_child_weight"],
                           colsample_bytree=xgheart.best_params_["colsample_bytree"],
                           eval_metric=xgheart.best_params_["eval_metric"])
xgs = xgboost.XGBRegressor(eta=xgskin.best_params_["eta"], max_depth=xgskin.best_params_["max_depth"],
                           min_child_weight=xgskin.best_params_["min_child_weight"],
                           colsample_bytree=xgskin.best_params_["colsample_bytree"],
                           eval_metric=xgskin.best_params_["eval_metric"])

while(iteration < 10):
    # Linear models
    #rh = LinearRegression(fit_intercept=regheart.best_params_["fit_intercept"])
    rh.fit(X1_train, y1_train)
    predictions = rh.predict(X1_test)
    rhmae.append(mean_absolute_error(y1_test, predictions))
    rhrmse.append(math.sqrt(mean_squared_error(y1_test, predictions)))
    #print("Linear heart MAE: ", mean_absolute_error(y1_test, predictions))
    #print("Linear heart RMSE: ", math.sqrt(mean_squared_error(y1_test, predictions)))

    #rs = LinearRegression(fit_intercept=regheart.best_params_["fit_intercept"])
    rs.fit(X2_train, y2_train)
    predictions = rs.predict(X2_test)
    rsmae.append(mean_absolute_error(y2_test, predictions))
    rsrmse.append(math.sqrt(mean_squared_error(y2_test, predictions)))
    #print("Linear skin MAE: ", mean_absolute_error(y2_test, predictions))
    #print("Linear skin RMSE: ", math.sqrt(mean_squared_error(y2_test, predictions)))

    # Rand Forest
    #fh = RandomForestRegressor(criterion=regheart_rf.best_params_["criterion"], max_depth=regheart_rf.best_params_["max_depth"], max_features=regheart_rf.best_params_["max_features"])
    fh.fit(X1_train, y1_train)
    fhpredictions = fh.predict(X1_test)
    fhmae.append(mean_absolute_error(y1_test, fhpredictions))
    fhrmse.append(math.sqrt(mean_squared_error(y1_test, fhpredictions)))
    fhmape.append(mean_absolute_percentage_error(y1_test, fhpredictions))
    #print("Forest heart MAE: ", mean_absolute_error(y1_test, fhpredictions))
    #print("Forest heart RMSE: ", math.sqrt(mean_squared_error(y1_test, fhpredictions)))

    #fs = RandomForestRegressor(criterion=regskin_rf.best_params_["criterion"], max_depth=regskin_rf.best_params_["max_depth"], max_features=regskin_rf.best_params_["max_features"])
    fs.fit(X2_train, y2_train)
    fspredictions = fs.predict(X2_test)
    fsmae.append(mean_absolute_error(y2_test, fspredictions))
    fsrmse.append(math.sqrt(mean_squared_error(y2_test, fspredictions)))
    fsmape.append(mean_absolute_percentage_error(y2_test, fspredictions))
    #print("Forest skin MAE: ", mean_absolute_error(y2_test, fspredictions))
    #print("Forest skin RMSE: ", math.sqrt(mean_squared_error(y2_test, fspredictions)))

    # XGB
    #xgh = xgboost.XGBRegressor(eta=xgheart.best_params_["eta"], max_depth=xgheart.best_params_["max_depth"], min_child_weight=xgheart.best_params_["min_child_weight"], colsample_bytree=xgheart.best_params_["colsample_bytree"], eval_metric=xgheart.best_params_["eval_metric"])
    xgh.fit(X1_train, y1_train)
    predictions = xgh.predict(X1_test)
    xhmae.append(mean_absolute_error(y1_test, predictions))
    xhrmse.append(math.sqrt(mean_squared_error(y1_test, predictions)))
    #print("XGB heart MAE: ", mean_absolute_error(y1_test, predictions))
    #print("XGB heart RMSE: ", math.sqrt(mean_squared_error(y1_test, predictions)))

    #xgs = xgboost.XGBRegressor(eta=xgskin.best_params_["eta"], max_depth=xgskin.best_params_["max_depth"], min_child_weight=xgskin.best_params_["min_child_weight"], colsample_bytree=xgskin.best_params_["colsample_bytree"], eval_metric=xgskin.best_params_["eval_metric"])
    xgs.fit(X2_train, y2_train)
    predictions = xgs.predict(X2_test)
    xsmae.append(mean_absolute_error(y2_test, predictions))
    xsrmse.append(math.sqrt(mean_squared_error(y2_test, predictions)))
    #print("XGB skin MAE: ", mean_absolute_error(y2_test, predictions))
    #print("XGB skin RMSE: ", math.sqrt(mean_squared_error(y2_test, predictions)))

    iteration += 1
print("Lin heart MAE: ", np.mean(rhmae))
print("Lin heart RMSE: ", np.mean(rhrmse))
print("Lin skin MAE: ", np.mean(rsmae))
print("Lin skin RMSE: ", np.mean(rsrmse))
print("RF heart MAE: ", np.mean(fhmae))
print("RF heart RMSE: ", np.mean(fhrmse))
print("RF skin MAE: ", np.mean(fsmae))
print("RF skin RMSE: ", np.mean(fsrmse))
print("RF heart MAPE: ", np.mean(fhmape))
print("RF skin MAPE: ", np.mean(fsmape))
print("XGB heart MAE: ", np.mean(xhmae))
print("XGB heart RMSE: ", np.mean(xhrmse))
print("XGB skin MAE: ", np.mean(xsmae))
print("XGB skin RMSE: ", np.mean(xsrmse))


################## Classification #################
heart_rate = dataset["Heart Rate"]
environment = pd.Series(range(len(heart_rate)))
for i in range(len(heart_rate)):
    if heart_rate[i] < 65:
        environment[i] = 0
    elif 65 <= heart_rate[i] < 80:
        environment[i] = 1
    elif heart_rate[i] >= 80:
        environment[i] = 2

print("0 class percentage", environment.value_counts()[0]/len(environment))
print("1 class percentage", environment.value_counts()[1]/len(environment))
print("2 class percentage", environment.value_counts()[2]/len(environment))

environment = list(environment)

dataset = pd.concat([dataset, environment], axis=1, ignore_index=True)
dataset.rename(columns={"0": "Environment"}, inplace=True)

X = dataset[
    ["Light", "Nr.particles0.3", "Nr.particles0.5", "Nr.particles1.0", "Temperature", "Humidity", "Pressure", "Sound",
     "IAQ"]]
# X = normalize(X, norm="l2")
y1 = environment
y2 = dataset["Skin Resistance"]
# Split dataset for train and test subsets
X1_train, X1_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.15)
X2_train, X2_test, y2_train, y2_test = train_test_split(X, y2, test_size=0.15)

lin_model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
lin_model.fit(X1_train, y1_train)
predictions = lin_model.predict(X1_test)
accuracy = accuracy_score(y1_test, predictions)
print("Accuracy: ", accuracy)
print(classification_report(y1_test, predictions, target_names=["Low risk", "Medium risk", "High risk"], digits=2))

class_model = RandomForestClassifier()
class_model.fit(X1_train, y1_train)
predictions = class_model.predict(X1_test)
accuracy = accuracy_score(y1_test, predictions)
print("Accuracy: ", accuracy)
print(classification_report(y1_test, predictions, target_names=["Low risk", "Medium risk", "High risk"], digits=2))

class_model_xgb = XGBClassifier()
class_model_xgb.fit(X1_train, y1_train)
predictions = class_model_xgb.predict(X1_test)
accuracy = accuracy_score(y1_test, predictions)
print("Accuracy: ", accuracy)
print(classification_report(y1_test, predictions, target_names=["Low risk", "Medium risk", "High risk"], digits=2))

feature_names = ["Light", "Nr.particles0.3", "Nr.particles0.5", "Nr.particles1.0", "Temperature", "Humidity",
                 "Pressure", "Sound", "IAQ"]
importances = class_model.feature_importances_
# plot feature importance
model_importances = pd.Series(importances, index=feature_names)

# Statistics
stat_data = pd.concat([dataset[
                           ["Light", "Nr.particles0.3", "Nr.particles0.5", "Nr.particles1.0", "Temperature", "Humidity",
                            "Pressure", "Sound", "IAQ"]], environment],
                      ignore_index=True, axis=1)
stat_data = stat_data.rename(columns={0: "Light", 1: "Nr.particles0.3", 2: "Nr.particles0.5", 3: "Nr.particles1.0",
                                      4: "Temperature", 5: "Humidity", 6: "Pressure", 7: "Sound", 8: "IAQ",
                                      9: "environment"})

print("zero ", stat_data.describe())

zero = stat_data.loc[stat_data["environment"] == 0]
print("zero ", zero.describe())

fig, axs = plt.subplots(1, 3)
fig.tight_layout(pad=2.5)

axs[0].grid(visible=True, which='major', axis='both')
axs[0].hist(zero["Temperature"], bins=30, density=True, color="green")
axs[0].set_xlabel("Temperature °C", fontsize=11)
axs[0].tick_params(axis='x', labelsize=11)
axs[0].tick_params(axis='y', labelsize=11)
axs[0].set_ylabel("Occurrences", fontsize=11)

axs[1].grid(visible=True, which='major', axis='both')
axs[1].hist(zero["Humidity"], bins=30, density=True, color="green")
axs[1].set_xlabel("Humidity", fontsize=11)
axs[1].tick_params(axis='x', labelsize=11)
axs[1].tick_params(axis='y', labelsize=11)
axs[1].set_ylabel("Occurrences", fontsize=11)

axs[2].grid(visible=True, which='major', axis='both')
axs[2].hist(zero["Pressure"], bins=30, density=True, color="green")
axs[2].set_xlabel("Pressure", fontsize=11)
axs[2].tick_params(axis='x', labelsize=11)
axs[2].tick_params(axis='y', labelsize=11)
axs[2].set_ylabel("Occurrences", fontsize=11)

plt.show()

columns = ["Skin Resistance", "Light", "Nr.particles0.3", "Nr.particles0.5",
           "Nr.particles1.0", "Temperature", "Humidity", "Pressure",
           "IAQ", "Sound", "Heart Rate"]

# Creating separate whisker plots for each variable
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 12))
fig.suptitle('Whisker Plots for Each Variable', fontsize=16)

for i, col in enumerate(columns):
    ax = axes[i // 4, i % 4]
    dataset.boxplot(column=col, ax=ax)
    ax.set_title(col)
    ax.set_ylabel('Values')

# Adjust layout to prevent clipping of titles
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Create a correlation matrix
correlation_matrix = np.corrcoef(dataset[["Skin Resistance", "Light", "Nr.particles0.3", "Nr.particles0.5",
                                          "Nr.particles1.0", "Temperature", "Humidity", "Pressure",
                                          "IAQ", "Sound", "Heart Rate"]], rowvar=False)

variable_names = ["Skin Resistance", "Light", "Nr.particles0.3", "Nr.particles0.5",
                  "Nr.particles1.0", "Temperature", "Humidity", "Pressure",
                  "IAQ", "Sound", "Heart Rate"]
# Plot the correlation matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.set(font_scale=1)  # Adjust font size if needed
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", square=True,
            xticklabels=variable_names, yticklabels=variable_names)
plt.tight_layout()
plt.show()

# Create 3D scatter plot with hue
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(stat_data["Pressure"], stat_data["Temperature"],
                     stat_data["Light"], c=stat_data["environment"],
                     cmap='brg', s=50, alpha=0.8)
ax.set_xlabel('Pressure (hPA)')
ax.set_ylabel('Temperature (°C)')
ax.set_zlabel('Light (lux)')
# Create custom legend handles and labels
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Low Risk'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Medium Risk'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='High Risk')]

# Set custom legend elements
ax.legend(handles=legend_elements, loc='upper right')

plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(stat_data["Humidity"], stat_data["IAQ"], stat_data["Sound"], c=stat_data["environment"],
                     cmap='brg', s=50, alpha=0.8)
ax.set_xlabel('Humidity (%)')
ax.set_ylabel('IAQ')
ax.set_zlabel('Sound (db)')
# Create custom legend handles and labels
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Low Risk'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Medium Risk'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='High Risk')]

# Set custom legend elements
ax.legend(handles=legend_elements, loc='upper right')

plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(stat_data["Nr.particles1.0"], stat_data["Nr.particles0.5"],
                     stat_data["Nr.particles0.3"], c=stat_data["environment"],
                     cmap='brg', s=50, alpha=0.8)
ax.set_xlabel('Nr.particles1.0')
ax.set_ylabel('Nr.particles0.5')
ax.set_zlabel('Nr.particles0.3')

# Create custom legend handles and labels
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Low Risk'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Medium Risk'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='High Risk')]

# Set custom legend elements
ax.legend(handles=legend_elements, loc='upper right')

plt.show()

########### Visualizations #########
variables = ['Sound', 'Light', 'Nr.particles0.3', 'Nr.particles0.5', 'Nr.particles1.0', 'Temperature', 'Humidity',
             'Pressure', 'IAQ']
bright_cmap = ["#00FF00", "#0000FF", "#FF0000"]

# Line histogram plot for all variables for all participants, including HR and SR
fig, axs = plt.subplots(4, 3)
sns.kdeplot(data=dataset['Sound'], color='green', ax=axs[0, 0])
axs[0, 0].set_ylabel("Density")
sns.kdeplot(data=dataset['Light'], color='green', ax=axs[0, 1])
axs[0, 1].set_ylabel("Density")
sns.kdeplot(data=dataset['Nr.particles0.3'], color='green', ax=axs[0, 2])
axs[0, 2].set_ylabel("Density")
sns.kdeplot(data=dataset['Nr.particles0.5'], color='green', ax=axs[1, 0])
axs[1, 0].set_ylabel("Density")
sns.kdeplot(data=dataset['Nr.particles1.0'], color='green', ax=axs[1, 1])
axs[1, 1].set_ylabel("Density")
sns.kdeplot(data=dataset['Temperature'], color='green', ax=axs[1, 2])
axs[1, 2].set_ylabel("Density")
sns.kdeplot(data=dataset['Humidity'], color='green', ax=axs[2, 0])
axs[2, 0].set_ylabel("Density")
sns.kdeplot(data=dataset['Pressure'], color='green', ax=axs[2, 1])
axs[2, 1].set_ylabel("Density")
sns.kdeplot(data=dataset['IAQ'], color='green', ax=axs[2, 2])
axs[2, 2].set_ylabel("Density")
sns.kdeplot(data=dataset['Heart Rate'], color='green', ax=axs[3, 0])
axs[3, 0].set_ylabel("Density")
sns.kdeplot(data=dataset['Skin Resistance'], color='green', ax=axs[3, 1])
axs[3, 1].set_ylabel("Density")
fig.tight_layout()
plt.show()

# Whisker plot for all variables for all participants, including HR and SR
fig, axs = plt.subplots(4, 3)
sns.boxplot(data=dataset, y='Sound', palette=bright_cmap, ax=axs[0, 0])
axs[0, 0].set_ylabel("Sound")
sns.boxplot(data=dataset, y='Light', palette=bright_cmap, ax=axs[0, 1])
axs[0, 1].set_ylabel("Light")
sns.boxplot(data=dataset, y='Nr.particles0.3', palette=bright_cmap, ax=axs[0, 2])
axs[0, 2].set_ylabel("Nr.particles0.3")
sns.boxplot(data=dataset, y='Nr.particles0.5', palette=bright_cmap, ax=axs[1, 0])
axs[1, 0].set_ylabel("Nr.particles0.5")
sns.boxplot(data=dataset, y='Nr.particles1.0', palette=bright_cmap, ax=axs[1, 1])
axs[1, 1].set_ylabel("Nr.particles1.0")
sns.boxplot(data=dataset, y='Temperature', palette=bright_cmap, ax=axs[1, 2])
axs[1, 2].set_ylabel("Temperature")
sns.boxplot(data=dataset, y='Humidity', palette=bright_cmap, ax=axs[2, 0])
axs[2, 0].set_ylabel("Humidity")
sns.boxplot(data=dataset, y='Pressure', palette=bright_cmap, ax=axs[2, 1])
axs[2, 1].set_ylabel("Pressure")
sns.boxplot(data=dataset, y='IAQ', palette=bright_cmap, ax=axs[2, 2])
axs[2, 2].set_ylabel("IAQ")
sns.boxplot(data=dataset, y='Heart Rate', palette=bright_cmap, ax=axs[3, 0])
axs[3, 0].set_ylabel("Heart Rate")
sns.boxplot(data=dataset, y='Skin Resistance', palette=bright_cmap, ax=axs[3, 1])
axs[3, 1].set_ylabel("Skin Resistance")
fig.tight_layout()
plt.show()

# Create a 3 by 3 grid of scatter plots
fig, axs = plt.subplots(3, 3, sharex=True)
sns.violinplot(data=dataset, x=environment, y='Sound', kind="violin", inner="quartile", palette=bright_cmap,
               ax=axs[0, 0])
axs[0, 0].set_ylabel("Sound")
sns.violinplot(data=dataset, x=environment, y='Light', kind="violin", inner="quartile", palette=bright_cmap,
               ax=axs[0, 1])
axs[0, 1].set_ylabel("Light")
sns.violinplot(data=dataset, x=environment, y='Nr.particles0.3', kind="violin", inner="quartile", palette=bright_cmap,
               ax=axs[0, 2])
axs[0, 2].set_ylabel("Nr.particles0.3")
sns.violinplot(data=dataset, x=environment, y='Nr.particles0.5', kind="violin", inner="quartile", palette=bright_cmap,
               ax=axs[1, 0])
axs[1, 0].set_ylabel("Nr.particles0.5")
sns.violinplot(data=dataset, x=environment, y='Nr.particles1.0', kind="violin", inner="quartile", palette=bright_cmap,
               ax=axs[1, 1])
axs[1, 1].set_ylabel("Nr.particles1.0")
sns.violinplot(data=dataset, x=environment, y='Temperature', kind="violin", inner="quartile", palette=bright_cmap,
               ax=axs[1, 2])
axs[1, 2].set_ylabel("Temperature")
sns.violinplot(data=dataset, x=environment, y='Humidity', kind="violin", inner="quartile", palette=bright_cmap,
               ax=axs[2, 0])
axs[2, 0].set_xlabel("Heart Conditions Risk")
axs[2, 0].set_ylabel("Humidity")
sns.violinplot(data=dataset, x=environment, y='Pressure', kind="violin", inner="quartile", palette=bright_cmap,
               ax=axs[2, 1])
axs[2, 1].set_xlabel("Heart Conditions Risk")
axs[2, 1].set_ylabel("Pressure")
sns.violinplot(data=dataset, x=environment, y='IAQ', kind="violin", inner="quartile", palette=bright_cmap, ax=axs[2, 2])
axs[2, 2].set_xlabel("Heart Conditions Risk")
axs[2, 2].set_ylabel("IAQ")

legend_labels = ["0 - low", "1 - medium", "2 - high"]
legend_colors = ['green', 'blue', 'red']
handles = [plt.Line2D([], [], color=color, label=label) for label, color in zip(legend_labels, legend_colors)]
fig.legend(handles=handles, labels=legend_labels, loc='center right', title='Heart Condition Risk', fancybox=True,
           shadow=True)

plt.show()


