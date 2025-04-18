from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

def knn_algorithm_smote(df, k=5, distance_metric='euclidean', use_smote=False):
    # Split the dataset into features and target
    x, y = df.select_dtypes(include=['number']).drop(columns=['ground_truth_labels'], errors='ignore'), df['ground_truth_labels']

    # Split the dataset into training and testing sets (80% train, 20% test)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42)

    # Apply SMOTE if specified (this balances the training set by generating synthetic samples)
    if use_smote:
        smote = SMOTE(random_state=42)
        x_train, y_train = smote.fit_resample(x_train, y_train)
    
    # Standardize the data (important for KNN)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Initialize and fit the k-NN classifier
    knn_clf = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
    knn_clf.fit(x_train, y_train)

    # Final test evaluation (prediction)
    y_test_pred = knn_clf.predict(x_test)

    # Calculate accuracy score
    test_accuracy = accuracy_score(y_test, y_test_pred)  # use the function correctly

    return test_accuracy, y_test, y_test_pred

#we can decide what it should return, for now it returns the accuracy score and the predictions
#the return will depend on what we want the result_baseline.csv to contain