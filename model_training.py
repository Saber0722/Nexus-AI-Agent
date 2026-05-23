import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv('path/to/your/dataset.csv')

# Separate features and target variable
X = df.drop(columns=['target_column'])
y = df['target_column']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verify the split
print(f"Training set features shape: {X_train.shape}")
print(f"Testing set features shape: {X_test.shape}")