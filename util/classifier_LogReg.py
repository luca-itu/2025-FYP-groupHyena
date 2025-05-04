from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def logistic_regression_classifier(df, use_smote=False):
    # Extract numeric features and target column
    X = df.select_dtypes(include=['number']).drop(columns=['ground_truth_labels'], errors='ignore')
    y = df['ground_truth_labels']

    # Train-validation-test split: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Apply SMOTE if requested
    if use_smote:
        minority_class = sum(y_train == 1)
        if minority_class < 2:
            raise ValueError("Not enough positive samples for SMOTE.")
        k_neighbors = min(minority_class - 1, 5)
        sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train logistic regression
    clf = LogisticRegression(class_weight='balanced', random_state=42)
    clf.fit(X_train_scaled, y_train)

    # Validation prediction and accuracy
    y_val_pred = clf.predict(X_val_scaled)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    # Test prediction and accuracy
    #y_test_pred = clf.predict(X_test_scaled)
    #test_accuracy = accuracy_score(y_test, y_test_pred)

    # Get probabilities for test set
    #y_test_probs = clf.predict_proba(X_test_scaled)

    #return val_accuracy, test_accuracy, y_test, y_test_pred, y_test_probs
    return y_val, y_val_pred, val_accuracy