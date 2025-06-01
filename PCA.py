from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

digits = load_digits()
X = digits.data
y = digits.target

n_components = 30
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)
X_reconstructed = pca.inverse_transform(X_pca)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

index = 0
original_image = X[index].reshape(8, 8)
reconstructed_image = X_reconstructed[index].reshape(8, 8)

fig, axs = plt.subplots(1, 2, figsize=(6, 3))
axs[0].imshow(original_image, cmap='gray')
axs[0].set_title("Original Image")
axs[0].axis('off')

axs[1].imshow(reconstructed_image, cmap='gray')
axs[1].set_title(f"PCA (n={n_components})")
axs[1].axis('off')
plt.tight_layout()
plt.show()


sample_index = 0
print("Predicted number:", y_pred[sample_index])
print("Actual number:   ", y_test[sample_index])

image = pca.inverse_transform(X_test[sample_index]).reshape(8, 8)
plt.imshow(image, cmap='gray')
plt.title(f"Predicted: {y_pred[sample_index]} / Actual: {y_test[sample_index]}")
plt.axis('off')
plt.show()
