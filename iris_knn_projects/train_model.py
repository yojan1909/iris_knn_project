import os 
import joblib 
import pandas as pd 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
def main(): 
# Create models folder if it does not exist 
    os.makedirs("models", exist_ok=True) 
    os.makedirs("data", exist_ok=True) 
# 1. Load dataset 
    iris = load_iris()  
    X = iris.data 
    y = iris.target 
# Save dataset to CSV for beginners to inspect 
    df = pd.DataFrame(X, columns=iris.feature_names) 
    df["target"] = y 
    df["species"] = df["target"].map({ 
    0: "setosa", 
    1: "versicolor", 
    2: "virginica" 
    }) 
    df.to_csv("data/iris.csv", index=False) 
    print("Dataset saved to data/iris.csv") 
# 2. Split data into train and test 
    X_train, X_test, y_train, y_test = train_test_split( 
        X, y, test_size=0.2, random_state=42, stratify=y 
    ) 
# 3. Scale the features 
    scaler = StandardScaler() 
    X_train_scaled = scaler.fit_transform(X_train) 
    X_test_scaled = scaler.transform(X_test) 
# 4. Create KNN model 
    model = KNeighborsClassifier(n_neighbors=3) 
# 5. Train model 
    model.fit(X_train_scaled, y_train) 
# 6. Make predictions 
    y_pred = model.predict(X_test_scaled) 
# 7. Evaluate model 
    accuracy = accuracy_score(y_test, y_pred) 
    print(f"Accuracy: {accuracy:.4f}") 
    print("\nConfusion Matrix:") 
    print(confusion_matrix(y_test, y_pred)) 
    print("\nClassification Report:") 
    print(classification_report(y_test, y_pred, target_names=iris.target_names)) 
# 8. Save model and scaler 
    joblib.dump(model, "models/knn_model.pkl") 
    joblib.dump(scaler, "models/scaler.pkl") 
    print("\nModel saved to models/knn_model.pkl") 
    print("Scaler saved to models/scaler.pkl") 
if __name__ == "__main__":
    main()
