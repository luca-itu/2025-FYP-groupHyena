from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def random_forest_classifier(df, use_smote=False):
    # Extract numeric features and target column
    X = df.select_dtypes(include=['number']).drop(columns=['ground_truth'], errors='ignore')
    y = df['ground_truth']

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

    clf = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)
    clf.fit(X_train, y_train)

    # predicted probabilities
    y_val_probs = clf.predict_proba(X_val)
    y_val_prob_1 = y_val_probs[:,1]

    # validation predictions with threshold
    threshold_val = 0.05
    y_val_pred = (y_val_prob_1>=threshold_val).astype(int)

    # compute validation accuracy
    val_accuracy = accuracy_score(y_val, y_val_pred)

    # test probabilities
    y_test_probs = clf.predict_proba(X_test)
    y_test_prob_1 = y_test_probs[:,1]
    
    # test predictions
    threshold_test = 0.01
    y_test_pred = (y_test_prob_1 >=threshold_test).astype(int)

    # compute test accuracy
    test_accuracy = accuracy_score(y_test, y_test_pred)

    return y_test, y_test_pred, test_accuracy, y_test_probs, y_val, y_val_pred, val_accuracy, y_val_probs
