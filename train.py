from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(f'Test accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f} ({len(X_test)} samples)')
    joblib.dump(model, Path(__file__).with_name('model.joblib'))

if __name__ == '__main__':
    train()
