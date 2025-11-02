import matplotlib.pyplot as plt

# Example data (replace these with your actual values)
epochs = list(range(1, 21))  # 20 epochs
train_loss = [3.4, 3.1, 2.9, 2.6, 2.4, 2.2, 2.0, 1.9, 1.8, 1.7,
               1.6, 1.5, 1.45, 1.4, 1.38, 1.35, 1.33, 1.30, 1.28, 1.25]

char_accuracy = [50, 58, 65, 69, 72, 75, 78, 80, 82, 84,
                 85, 86, 87, 88, 89, 90, 91, 91.3, 91.5, 91.64]

# Create figure
plt.figure(figsize=(9, 6))

# Plot Loss
plt.plot(epochs, train_loss, label='Training Loss', color='red', marker='o', linewidth=2)

# Plot Accuracy
plt.plot(epochs, char_accuracy, label='Character Accuracy (%)', color='green', marker='s', linewidth=2)

# Add labels and title
plt.title("Model Training Progress Over Epochs", fontsize=14, fontweight='bold')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()

# Display the plot
plt.show()
