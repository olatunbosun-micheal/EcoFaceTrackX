from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,classification_report

# Sample ground truth data and predictions
# Replace these lists with actual data from your attendance system
# Corrected Ground Truth Labels and Model Predictions
actual_attendance = ['present',  'unknown', 'present','unknown', 'present','unknown','unknown','present','present','present','present','unknown','present','present','present','unknown','present']  # Ground truth labels
predicted_attendance = ['present',   'unknown', 'present','unknown' ,'present', 'unknown','present','present','present','unknown','present','unknown','present','present','present','unknown','present']  # Model predictions


# Calculate accuracy, precision, recall, and F1-score
accuracy = accuracy_score(actual_attendance, predicted_attendance)
precision = precision_score(actual_attendance, predicted_attendance, pos_label='present')
recall = recall_score(actual_attendance, predicted_attendance, pos_label='present')
f1 = f1_score(actual_attendance, predicted_attendance, pos_label='present')

# Print the metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

report = classification_report(actual_attendance, predicted_attendance)
print(report)