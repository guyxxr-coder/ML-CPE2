import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

class SVMModelTrainer:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        self.scaler = StandardScaler()
        self.model = SVC(kernel=kernel, C=C, gamma=gamma, class_weight='balanced', random_state=42)

    def train_and_evaluate(self, X, y, split_ratio=0.8):
        split_idx = int(len(X) * split_ratio)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Feature Scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train & Predict
        self.model.fit(X_train_scaled, y_train)
        y_pred = self.model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)
        
        # วาดและบันทึกรูป Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Pred Down (0)', 'Pred Up (1)'],
                    yticklabels=['Actual Down (0)', 'Actual Up (1)'])
        plt.title('Confusion Matrix - SVM Model')
        plt.ylabel('Actual Class')
        plt.xlabel('Predicted Class')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300) # บันทึกเป็นไฟล์ภาพ
        plt.show() # แสดงหน้าต่างรูปภาพ
        
        return acc, report, y_test, y_pred