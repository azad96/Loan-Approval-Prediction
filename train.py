import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, KFold


df = pd.read_csv('loan_data.csv')

# Make all string columns lowercase and replace spaces with underscores
for column in df.select_dtypes(include=['object']).columns:
    df[column] = df[column].str.lower().str.replace(' ', '_')

df.rename(columns={'loan_status': 'y'}, inplace=True)

numerical = ["person_age", "person_income", "person_emp_exp", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length", "credit_score"]
categorical = ["person_gender", "person_education", "person_home_ownership", "loan_intent", "previous_loan_defaults_on_file"]

# Create histograms for all numerical variables
fig, axes = plt.subplots(4, 2, figsize=(15, 10))
axes = axes.ravel()

for idx, col in enumerate(numerical):
    sns.histplot(data=df, x=col, ax=axes[idx])
    axes[idx].set_title(f'Distribution of {col}')

plt.tight_layout()
plt.show()

# Create countplots for categorical variables
fig, axes = plt.subplots(3, 2, figsize=(15, 10))
axes = axes.ravel()

for idx, col in enumerate(categorical):
    sns.countplot(data=df, x=col, ax=axes[idx])
    axes[idx].set_title(f'Count of {col}')
    axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Distribution of target variable
plt.figure(figsize=(4, 3))
sns.countplot(data=df, x='y')
plt.title('Distribution of Loan Approval Status')
plt.show()

# Calculate class imbalance
print("Class distribution:")
print(df['y'].value_counts(normalize=True))

# Create correlation matrix heatmap
plt.figure(figsize=(8, 4))
correlation_matrix = df[numerical].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numerical Variables')
plt.show()

# Remove highly correlated features
df = df.drop(columns=['person_emp_exp', 'cb_person_cred_hist_length'])
numerical.remove('person_emp_exp')
numerical.remove('cb_person_cred_hist_length')

# Split the data into training, validation, and test sets
seed = 42
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=seed)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=seed)
print(len(df_train), len(df_val), len(df_test))

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.y.values
y_val = df_val.y.values
y_test = df_test.y.values

del df_train['y']
del df_val['y']
del df_test['y']

# Check feature importance of numerical variables.
scores = []
for c in numerical:
  auc = roc_auc_score(y_train, df_train[c])
  if auc < 0.5:
    auc = roc_auc_score(y_train, -df_train[c])
  scores.append((c, auc))

scores.sort(key=lambda x: x[1], reverse=True)
for c, score in scores:
  print(c, score)

plt.figure(figsize=(5, 5))
fpr, tpr, _ = roc_curve(y_train, df_train["loan_percent_income"])
plt.plot(fpr, tpr, label='loanpercent_income')
fpr, tpr, _ = roc_curve(y_train, df_train["loan_int_rate"])
plt.plot(fpr, tpr, label='loan_int_rate')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.legend()
plt.show()

def mutual_info_y_score(df_column):
  return mutual_info_score(df_column, y_train)

mutual_info_scores = df_train[categorical].apply(mutual_info_y_score)
mutual_info_scores.sort_values(ascending=False).to_frame(name='Mutual Information')

def train(df_train, y_train, C=1.0):
  dv = DictVectorizer(sparse=False)

  train_dict = df_train.to_dict(orient='records')
  X_train = dv.fit_transform(train_dict)

  model = LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=1)
  model.fit(X_train, y_train)

  return model, dv

def predict(model, df_val, dv):
  val_dict = df_val.to_dict(orient='records')
  X_val = dv.transform(val_dict)
  y_pred = model.predict_proba(X_val)[:,1]
  return y_pred

# Training
model, dv = train(df_train, y_train)

# Validation
y_pred = predict(model, df_val, dv)
roc_auc_score(y_val, y_pred)

def PR_dataframe(y_val, y_pred):
    scores = []
    thresholds = np.linspace(0, 1, 101)

    for t in thresholds:
        actual_positive = (y_val == 1)
        actual_negative = (y_val == 0)

        predict_positive = (y_pred >= t)
        predict_negative = (y_pred < t)

        tp = (predict_positive & actual_positive).sum()
        tn = (predict_negative & actual_negative).sum()

        fp = (predict_positive & actual_negative).sum()
        fn = (predict_negative & actual_positive).sum()

        scores.append((t, tp, fp, fn, tn))

    columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
    df_scores = pd.DataFrame(scores, columns=columns)

    df_scores['precision'] = df_scores.tp / (df_scores.tp + df_scores.fp)
    df_scores['recall'] = df_scores.tp / (df_scores.tp + df_scores.fn)

    return df_scores

df_scores = PR_dataframe(y_val, y_pred)

plt.plot(df_scores.threshold, df_scores['precision'], label='precision')
plt.plot(df_scores.threshold, df_scores['recall'], label='recall')
plt.vlines(0.327, 0, 1, color='grey', linestyle='--', alpha=0.5)
plt.legend()
plt.show()

df_scores['f1'] = 2 * df_scores['precision'] * df_scores['recall'] / (df_scores['precision'] + df_scores['recall'])

df_scores.loc[df_scores.f1.argmax()]

plt.plot(df_scores.threshold, df_scores['f1'])
plt.vlines(0.32, 0.0, 0.6, color='grey', linestyle='--', alpha=0.5)
plt.xticks(np.linspace(0, 1, 11))
plt.show()

kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
scores = []

for train_idx, val_idx in kfold.split(df_full_train):
  df_train = df_full_train.iloc[train_idx]
  df_val = df_full_train.iloc[val_idx]

  y_train = df_train.y.values
  y_val = df_val.y.values

  model, dv = train(df_train, y_train)
  y_pred = predict(model, df_val, dv)

  auc = roc_auc_score(y_val, y_pred)
  scores.append(auc)

print(np.mean(scores), np.std(scores))

C_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
for C in C_values:
  kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
  scores = []

  for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.y.values
    y_val = df_val.y.values

    model, dv = train(df_train, y_train, C=C)
    y_pred = predict(model, df_val, dv)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

  print(f'C: {C}, mean score: {np.mean(scores)}, std: {np.std(scores)}')

# Training the final model with the best C value of 0.01
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
scores = []

for train_idx, val_idx in kfold.split(df_full_train):
  df_train = df_full_train.iloc[train_idx]
  df_val = df_full_train.iloc[val_idx]

  y_train = df_train.y.values
  y_val = df_val.y.values

  model, dv = train(df_train, y_train, C=0.01)
  y_pred = predict(model, df_val, dv)

  auc = roc_auc_score(y_val, y_pred)
  scores.append(auc)

print(np.mean(scores), np.std(scores))

# Save the model
model_name = 'model.bin'
with open(model_name, 'wb') as f_out:
  pickle.dump((dv, model), f_out)

print(f'Model is saved to {model_name}')
