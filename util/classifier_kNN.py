from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

def knn_algorithm_smote(df, k=5, distance_metric='euclidean', use_smote=False):
    # Split the dataset into features and target
    x, y = df.select_dtypes(include=['number']).drop(columns=['ground_truth_labels'], errors='ignore'), df['ground_truth_labels']

    # Split the dataset into training, validation and testing sets (70% train, 15% test, 15 % val)
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, shuffle=True, random_state=42)

    # Apply SMOTE if specified, only on the training set
    if use_smote:
        smote = SMOTE(random_state=42,  k_neighbors=2)
        x_train, y_train = smote.fit_resample(x_train, y_train)
    
    # Standardize the data (important for KNN)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)  # Scales all numeric columns
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    # Initialize and fit the k-NN classifier
    knn_clf = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
    knn_clf.fit(x_train_scaled, y_train)

    # Validation

    y_val_pred = knn_clf.predict(x_val_scaled)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    # Final test evaluation (prediction)
    y_test_pred = knn_clf.predict(x_test_scaled)

    # Calculate accuracy score
    test_accuracy = accuracy_score(y_test, y_test_pred)  

    return val_accuracy, test_accuracy, y_test, y_test_pred

#we can decide what it should return, for now it returns the accuracy score and the predictions
#the return will depend on what we want the result_baseline.csv to contain