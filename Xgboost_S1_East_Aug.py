#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 02:12:50 2025

@author: alexporcayo
"""

import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,accuracy_score,classification_report
)
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, KFold
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.preprocessing import label_binarize

RDN=100


df = pd.read_csv("/Users/alexporcayo/Documents/python/EAST_DATA721.csv")


FN=['ApproachSpeed','Arrival_Rate5min', 'Arrival_Rate10min',
       'Arrival_Rate15min', 'Arrival_Rate30min', 'Arrival_Rate60min',
       'Departure_Rate5min', 'Departure_Rate10min', 'Departure_Rate15min',
       'Departure_Rate30min', 'Departure_Rate60min', 'Cross_Rate5min',
       'Cross_Rate10min', 'Cross_Rate15min', 'Cross_Rate30min',
       'Cross_Rate60min', 'FWD_H_56CNT',
       'BHD_H_56CNT',
       'windspeed', 'winddir', 'visibility', 'RAMP_1', 'RAMP_2', 'RAMP_3', 'RAMP_4', 'RAMP_5',
       'RAMP_6', 'Weather_clear-day',
       'Weather_clear-night', 'Weather_cloudy', 'Weather_fog',
       'Weather_partly-cloudy-day', 'Weather_partly-cloudy-night',
       'Weather_rain', 'Weather_wind', 'A20N', 'A21N', 'A319', 'A320', 'A321',
       'A332', 'A333', 'A339', 'A359', 'A35K', 'B38M', 'B39M', 'B712', 'B737',
       'B738', 'B739', 'B752', 'B753', 'B763', 'B764', 'B772', 'B77W', 'B788',
       'BCS1', 'BCS3', 'C208', 'C56X', 'CRJ2', 'CRJ7', 'CRJ9', 'E170', 'E190',
       'E75L', 'E75S', 'LJ75', 'PC12',  '1STEPFRONTFLAG', '1STEPBEHINDFLAG']

# Separate the dataset into features (X) and target (y)
X = df[FN]
y = df['East_Exit_Flag']

# # Split the data into training and testing sets (80% training, 20% testing)
X_train, X_testpre, y_train, y_testpre = train_test_split(
     X, y, test_size=0.20, stratify=y,random_state=RDN
 )


X_test, X_val, y_test, y_val = train_test_split(
     X_testpre, y_testpre, test_size=0.50,stratify=y_testpre,random_state=RDN
 )




param_test1 = {
  "n_estimators": randint(100, 500),

  # learning rate η
  "learning_rate": uniform(0.001, 0.20),   
  # max tree depth
  "max_depth": randint(3, 8),

  # subsample of rows per tree
  "subsample": uniform(0.5, 0.5),          

  # subsample of features per tree
  "colsample_bytree": uniform(0.5, 0.5),    

  # regularization
  "gamma": uniform(0, 8),
  "reg_alpha": uniform(0, 1),
  "reg_lambda": uniform(0, 10),

  # minimum child weight
  "min_child_weight": randint(1, 10)
}



# # # Split the data into training and testing sets (80% training, 20% testing)
# X_train, X_val, y_train, y_val = train_test_split(
#      X, y, test_size=0.1,stratify=y,random_state=RDN
#  )


gsearch1 = RandomizedSearchCV(estimator = XGBClassifier(num_class=3,learning_rate =0.12, max_depth=6,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=RDN), 
 param_distributions=param_test1,n_iter=50, scoring='accuracy',n_jobs=4, cv=5)





gsearch1.fit(X_train,y_train,eval_set=[(X_val,y_val)],verbose=False)
score = gsearch1.score(X_test, y_test)

#print('Accuracy:{}, feature importance:{}'.format(score,gsearch1.best_estimator_.feature_importances_))
#print(gsearch1.best_params_)


y_bin = label_binarize(y, classes=[0,1,2])

yb_test= label_binarize(y_test, classes=[0,1,2])

y_pred   = gsearch1.best_estimator_.predict(X_test)
y_proba  = gsearch1.best_estimator_.predict_proba(X_test)  


model = gsearch1.best_estimator_
NAMES=['B7','B11','B13']


fpr = dict(); tpr = dict(); roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(yb_test[:, i], y_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
# micro‑average
fpr["micro"], tpr["micro"], _ = roc_curve(
    yb_test.ravel(), y_proba.ravel()
)
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure(figsize=(6,5))
for i in range(3):
    plt.plot(fpr[i], tpr[i], lw=1.5,
             label=f"{NAMES[i]} (AUC={roc_auc[i]:.2f})")
plt.plot(fpr["micro"], tpr["micro"],
         linestyle='--', label=f"micro‑avg (AUC={roc_auc['micro']:.2f})")
plt.plot([0,1], [0,1], color='gray', lw=1, linestyle='--')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title(" East Flow Taxi Exit ROC AUC XGBoost")
plt.legend(loc="lower right")
plt.grid(True)
#plt.show()
plt.savefig('EF_XG_TEXIT_ROC_AUC.jpg', dpi=800)
#plt.show()





precision = dict(); recall = dict(); pr_auc = dict()
for i in range(3):
    precision[i], recall[i], _ = precision_recall_curve(
        yb_test[:, i], y_proba[:, i]
    )
    pr_auc[i] = auc(recall[i], precision[i])
# micro
precision["micro"], recall["micro"], _ = precision_recall_curve(
    yb_test.ravel(), y_proba.ravel()
)
pr_auc["micro"] = auc(recall["micro"], precision["micro"])


plt.figure(figsize=(6,5))
for i in range(3):
    plt.plot(recall[i], precision[i], lw=1.5,
             label=f"{NAMES[i]} (AUC={pr_auc[i]:.2f})")
plt.plot(recall["micro"], precision["micro"],
         linestyle='--', label=f"micro‑avg (AUC={pr_auc['micro']:.2f})")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("East Flow Taxi Exit Precision–Recall XGBoost")
plt.legend(loc="lower left")
plt.grid(True)
#plt.show()
plt.savefig('EF_XG_TEXIT_PR.jpg', dpi=800)
#plt.show()






cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['B7','B11','B13'])
disp.plot(cmap=plt.cm.Blues)
plt.title("East Flow Taxi Exit Confusion Matrix XGBoost")
#plt.show()
plt.savefig('EF_XG_TEXIT_CMatrix.jpg', dpi=800)
# plt.show()
# print()
# print()
#print(classification_report(y_test, y_pred))



# # # # Initialize the XGBoost classifier for multi-class classification:
# # # # - objective='multi:softmax' tells XGBoost to predict class labels directly.
# # # # - num_class=3 indicates that there are 3 classes.
# # # # - eval_metric='mlogloss' is the evaluation metric for multi-class classification.
# # # # - use_label_encoder=False avoids warnings from recent XGBoost versions.
# # # model = XGBClassifier(
# # #     objective='multi:softmax',
# # #     num_class=3,
# # #     eval_metric='mlogloss',
# # #     use_label_encoder=False,
# # #     random_state=RDN
# # # )

# # # model.fit(X_train,y_train,eval_set=[(X_val,y_val)],verbose=False)



# # # # 5. Make predictions
# # # y_pred_train = model.predict(X_train)
# # # y_pred_val   = model.predict(X_val)
# # # y_pred_test  = model.predict(X_test)

# # # # 6. Compute and print accuracy on each split

# # # print("TRAINING REPORT:")
# # # print()
# # # print(classification_report(y_train, y_pred_train,digits=4))
# # # print()

# # # print("VALIDATION REPORT:")
# # # print()
# # # print(classification_report(y_val, y_pred_val,digits=4))
# # # print()


print("TEST REPORT:")
print()
print(classification_report(y_test, y_pred,digits=5))
print()

# # importance_dict = dict(zip(feature_names, gsearch1.best_estimator_.feature_importances_))


# # d_sorted_by_value = dict(sorted(importance_dict.items(), key=lambda kv: kv[1]))

# # categories = list(d_sorted_by_value.keys())
# # counts     = list(d_sorted_by_value.values())

# # # Plot horizontal bars
# # plt.figure(figsize=(8,8))
# # plt.barh(categories, counts, edgecolor='black')
# # plt.xlabel('Weight')
# # plt.ylabel('Features')
# # plt.title('Taxi Cross Feature Importance of West Flow')
# # plt.tight_layout()
# # #plt.show()
# # plt.savefig('WF_XG_TC_FEAT.jpg', dpi=800)
# # plt.show()
# # # print()
# # # print('FEATURES IMPORTANCE:')

# # # for feature, importance in importance_dict.items():
# # #     print(f"  {feature}: {importance:.4f}")



# # # print()
# # # print('FEATURES IMPORTANCE:')

# # # for feature, importance in importance_dict.items():
# # #     print(f"  {feature}: {importance:.4f}")


# Create a TreeExplainer and compute SHAP values
explainer = shap.TreeExplainer(model)
# shap_values is a list with one array per class, each array shape (n_samples, n_features)
shap_values = explainer.shap_values(X_test)



# === 3. Global feature importance (bar plot) ===
# This shows the mean(|SHAP|) across all classes and samples
shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar",
    feature_names=X_test.columns,
    class_names=['B7','B11','B13'],
    plot_size=(9, 6),  # controls figure size inside SHAP
    show=False          # don't auto-show; lets us adjust & save cleanly
)

fig = plt.gcf()
# give extra room for the long x-axis label and tick labels, if any
fig.subplots_adjust(bottom=0.25, left=0.25)  # tweak if still tight

# Optional: make the axis label explicit and spaced from the axis
plt.xlabel("mean(|SHAP value|) — average impact on model output magnitude", labelpad=12)

# Save without cutting anything off
plt.savefig("EF_XG_TEXIT_SHAP.png", dpi=600, bbox_inches="tight", pad_inches=0.3)
plt.close(fig)

# # === 4. Class-specific summary plots ===
# for i, class_name in enumerate(['E1','E3','E5','C','D','End-Around']):
#     shap.summary_plot(
#         shap_values[i], 
#         X_test, 
#         feature_names=X_test.columns,
#         show=False
#     )
#     plt.title(f"SHAP summary for class {class_name}")
#     plt.tight_layout()
#     plt.show()

# # === 5. (Optional) Dependence plot for your top feature ===
# # First compute mean absolute SHAP for each feature:
# mean_abs = np.mean([np.abs(cls).mean(0) for cls in shap_values], axis=0)
# top_feat = X_test.columns[np.argmax(mean_abs)]
# # Then:
# shap.dependence_plot(
#     top_feat, 
#     shap_values,   # shap_values can be the full list; SHAP will pick the right slices
#     X_test, 
#     feature_names=X_test.columns
# )


