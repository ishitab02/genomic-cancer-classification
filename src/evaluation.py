import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# classification report 
def evaluate_model(y_true, y_pred, output_dir="outputs", model_name="model"):

        accuracy = accuracy_score(y_true, y_pred)
        print(f"\nAccuracy of {model_name}: {accuracy:.4f}")

        report = classification_report(y_true, y_pred)
        print(f"Classification Report for {model_name}:")
        print(report)

# saving classification report as .txt file
        report_path = os.path.join(output_dir, f"classification_report_{model_name}.txt")
        with open(report_path, "w") as f:
                f.write(f"Classification Report for {model_name}:\n")
                f.write(report)

 # saving confusion matrix as .png file
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix for {model_name}")
        plt.tight_layout()
        cm_path = os.path.join(output_dir, f"confusion_matrix_{model_name}.png")
        plt.savefig(cm_path)
        plt.close()

        return accuracy