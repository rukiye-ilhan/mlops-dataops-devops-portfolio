import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, roc_auc_score

#load data and cleaning
print("Veri yükleniyor ve temizleniyor...")

# mange the file path according to you
try:
    df = pd.read_csv('data/credit_risk_dataset.csv')
except FileNotFoundError:
    df = pd.read_csv('C:/Users/user/Desktop/mlops-dataops-devops-portfolio/01-future-mlops-credit-risk-prediction-project/data/credit_risk_dataset.csv')

# --- filterig invalid value ---
# Remove records where age is > 100 OR employment length is > 60 years.
invalid_filter = (df['person_age'] > 100) | (df['person_emp_length'] > 60)
df = df[~invalid_filter].copy()

#  (Imputation)
# Fill missing employment length with the median
emp_median = df['person_emp_length'].median()
df['person_emp_length'] = df['person_emp_length'].fillna(emp_median)

# Fill missing loan interest rates with the mean based on Loan Grade
df['loan_int_rate'] = df['loan_int_rate'].fillna(
    df.groupby('loan_grade')['loan_int_rate'].transform('mean')
)

print(f"Temizlik sonrası eksik değer sayısı:\n{df.isnull().sum().sum()}")
print(f"Final satır sayısı: {len(df)}")

# Save the cleaned data (Optional)
df.to_csv('credit_risk_cleaned.csv', index=False)


#FEATURE ENGINEERING & ENCODING
print("Performing Feature Engineering...")

# Map 'Y' and 'N' to 1 and 0 for default history
df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'Y': 1, 'N': 0})

# Map Loan Grades to numbers (Risk ranking)
grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
df['loan_grade'] = df['loan_grade'].map(grade_map)

# Convert categorical variables to Dummy variables (One-Hot Encoding)
# drop_first=True prevents multicollinearity
df = pd.get_dummies(df, columns=['person_home_ownership', 'loan_intent'], drop_first=True)

# Convert boolean (True/False) columns to integers (1/0)
bool_cols = df.select_dtypes(include=['bool']).columns
df[bool_cols] = df[bool_cols].astype(int)

#Save the processed data ready for training
df.to_csv('credit_risk_ready.csv', index=False)
print("Success: 'credit_risk_ready.csv' is created!")

#TRAIN-TEST SPLIT
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Use stratify=y to ensure balanced classes in split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# MLFLOW AND MODEL TRAINING (PIPELINE)

# Set the experiment name in MLflow
mlflow.set_experiment("Credit_Risk_Full_Pipeline")

print("\nMLflow deneyi başlıyor... Modeller karşılaştırılıyor...")

# Dictionary of models (For reference, strictly focusing on RF below)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42)
}

# Define Stratified K-Fold for Cross Validation

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- MLFLOW START RUN ---
# We will log the final model (Random Forest) in detail.
with mlflow.start_run(run_name="Random_Forest_Final_Model"):
    
    # 1. Create Pipeline
    # Combine Scaler and Model into a single pipeline object
    final_model = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)
    
    my_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', final_model)
    ])
    
   # Log parameters to MLflow
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("n_estimators", 100)

   # 2. Cross Validation (Stratified K-Fold)
    print("Running Stratified K-Fold CV...")
    cv_results = cross_validate(my_pipeline, X_train, y_train, cv=skf, scoring=('accuracy', 'recall', 'roc_auc'))
    
    mean_acc = np.mean(cv_results['test_accuracy'])
    mean_recall = np.mean(cv_results['test_recall'])
    mean_roc_auc = np.mean(cv_results['test_roc_auc'])

    print(f"Mean Accuracy: % {mean_acc * 100:.2f}")
    print(f"Mean Recall: % {mean_recall * 100:.2f}")
    print(f"Mean ROC-AUC: % {mean_roc_auc * 100:.2f}")

    # Log CV results as metrics to MLflow
    mlflow.log_metric("cv_mean_accuracy", mean_acc)
    mlflow.log_metric("cv_mean_recall", mean_recall)
    mlflow.log_metric("cv_mean_roc_auc", mean_roc_auc)

    # 3. Final Model Training (on full training set)
    print("\nTraining final model on full training set...")
    my_pipeline.fit(X_train, y_train)

   # 4. Test Set Predictions
    y_pred = my_pipeline.predict(X_test)
    y_prob = my_pipeline.predict_proba(X_test)[:, 1]

    # Test Skorları
    test_acc = accuracy_score(y_test, y_pred)
    test_roc = roc_auc_score(y_test, y_prob)
    
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

   # Log test metrics to MLflow
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_roc_auc", test_roc)

   # 5. Plot and Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['0: Paid', '1: Default'], 
                yticklabels=['0: Paid', '1: Default'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    # Upload the image as an "Artifact" to MLflow
    mlflow.log_artifact("confusion_matrix.png")
    print("Confusion Matrix uploaded to MLflow..")
    
    # 6. Save the Model (Both local and MLflow)
    # Local save
    joblib.dump(my_pipeline, 'credit_risk_model.joblib')
    # MLflow save (Saving as Pipeline)
    mlflow.sklearn.log_model(my_pipeline, "model_pipeline")
    
    print("Model and Pipeline saved successfully!")

print("\nScript completed.")