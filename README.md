# Predicting BACE Ki

## Introduction

This technical documentation provides a comprehensive approach to predicting BACE (Beta-site amyloid precursor protein cleaving enzyme) Ki values. The process involves data processing, molecular descriptor generation, and machine learning techniques. The goal is to understand the inhibitory activity of compounds against BACE enzymes, particularly BACE1, and develop predictive models for BACE Ki values associated with Alzheimer's disease.

## 1. Reading and Selecting Features

### 1.1 Reading the BACE Dataset
```python
import pandas as pd

# Read the Excel file into a DataFrame
bace_dataframe = pd.read_excel('bace_ki.xlsx')
```

### 1.2 Selecting Features and Saving to CSV
```python
# Select specific features of interest
bace_selected_features = bace_dataframe[['BindingDB Reactant_set_id', 'Ligand SMILES', 'BindingDB MonomerID', 'BindingDB Ligand Name', 'Target Name', 'Ki (nM)']]

# Save the selected features to a CSV file
bace_selected_features.to_csv('bace_selected_features.csv', index=False)
```

## 2. Data Cleaning

### 2.1 Removing Rows with '>'
```python
# Identify and remove rows with '>' in the 'Ki (nM)' column
index_sum = []

for index, row in bace_selected_features.iterrows():
    convert_ki_to_string = str(row['Ki (nM)'])
    
    if '>' in convert_ki_to_string:
        print(row['Ki (nM)'], index)
        index_sum.append(index)

# Drop rows with '>' from the DataFrame
bace_selected_features.drop(bace_selected_features.index[index_sum], inplace=True)
```

### 2.2 Saving Cleaned Data
```python
# Save the cleaned data to a new CSV file
bace_selected_features.to_csv("clean_bace_selected_features.csv", index=False)
```

## 3. Data Analysis and Visualization

### 3.1 Reading Cleaned Data
```python
# Read the cleaned data into a new DataFrame
clean_bace_selected_features = pd.read_csv("clean_bace_selected_features.csv")
```

### 3.2 Saving Ki Only (Unscaled)
```python
# Extract and save the 'Ki (nM)' column to a CSV file
bace_ki = clean_bace_selected_features['Ki (nM)']
bace_ki.to_csv('base_ki_only(unscaled).csv', index=False)
```

### 3.3 Descriptive Statistics
```python
# Display descriptive statistics for the 'Ki (nM)' column
pd.to_numeric(bace_ki).describe()
```

### 3.4 Shapiro-Wilk Test for Normality
```python
# Perform the Shapiro-Wilk test for normality
from scipy.stats import shapiro

stat, p = shapiro(bace_ki)
if p > 0.05:
    print(f'The data is likely Gaussian (p={p:.3f})')
else:
    print(f'The data is not likely Gaussian (p={p:.3f})')
```

# BACE Inhibitors Descriptor Generation

This section focuses on preparing and generating molecular descriptors for a dataset of BACE inhibitors.

## 1. Data Cleaning and Deduplication
```python
import pandas as pd

# Load the BACE inhibitors dataset
bace_inhibitors = pd.read_csv('clean_bace_selected_features.csv')

# Check for duplicate entries based on 'BindingDB MonomerID'
duplicated_entries = bace_inhibitors['BindingDB MonomerID'].duplicated().sum()

# Drop duplicate entries based on 'BindingDB MonomerID'
bace_inhibitors = bace_inhibitors.drop_duplicates(subset=['BindingDB MonomerID'], keep='first')

# Save the cleaned dataset
bace_inhibitors.to_csv('clean_bace_inhibitors.csv', index=False)
```

## 2. Descriptor Generation using RDKit

```python
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

# Check canonicity of the ligand SMILES notation using RDKit
def generate_canonical_smiles(mol_smiles):
    mols = [Chem.MolFromSmiles(mol_smile) for mol_smile in mol_smiles]
    canonical_smiles = [Chem.MolToSmiles(mol) for mol in mols] 
    return canonical_smiles 

# Generate canonical SMILES
canonical_smile = generate_canonical_smiles(bace_inhibitors['Ligand SMILES'])
bace_inhibitors['Ligand SMILES'] = canonical_smile

# Save the dataset with canonical SMILES
bace_inhibitors.to_csv('clean_bace_inhibitors.csv', index=False)

# Check for duplicated SMILES
duplicated_smiles = bace_inhibitors[bace_inhibitors['Ligand SMILES'].duplicated()]['Ligand SMILES'].values
len(duplicated_smiles)
```

## 3. Descriptor Generation using RDKit

### 3.1 RDKit Descriptors Calculation

```python
from rdkit.Chem import AllChem

# Generate RDKit descriptors
def generate_rdkit_descriptors(smile_list):
    # Initialize RDKit descriptor calculator
    rdkit_descriptor_names = [descriptor[0] for descriptor in Descriptors._descList]
    rdkit_descriptors = MoleculeDescriptors.MolecularDescriptorCalculator(rdkit_descriptor_names)
    
    # Convert SMILES to RDKit molecules
    ligands = [Chem.MolFromSmiles(ligand_smile) for ligand_smile in smile_list]
    
    # Calculate descriptors for each ligand
    ligand_rdkit_descriptors = [rdkit_descriptors.CalcDescriptors(Chem.AddHs(ligand)) for ligand in ligands]
    
    return ligand_rdkit_descriptors, rdkit_descriptor_names

# Apply RDKit descriptors generation to the dataset
rdkit_descriptor_values, rdkit_descriptor_names = generate_rdkit_descriptors(bace_inhibitors['Ligand SMILES'])

# Create a DataFrame for RDKit descriptors
descriptor_df = pd.DataFrame(rdkit_descriptor_values, columns=rdkit_descriptor_names)

# Combine RDKit descriptors with the original dataset
combined_bace_inhibitors = pd.concat([bace_inhibitors, descriptor_df], axis=1)

# Save the combined dataset with RDKit descriptors
combined_bace_inhibitors.to_csv('combined_bace_inhibitors_ki_and_rdkit_descriptors.csv', index=False)
combined_bace_inhibitors.shape
```

### 3.2 Mordred Descriptors Calculation

```python
from mordred import Calculator, descriptors

# Generate Mordred descriptors
def generate_mordred_descriptors(smile_molecules):
    # Generate RDKit molecules from SMILES
    ligands = [Chem.MolFromSmiles(smile) for smile in smile_molecules]
    
    # Generate Mordred descriptors
    calc = Calculator(descriptors, ignore_3D=False)
    mordred_descriptors = calc.pandas(ligands)
    
    return mord
```

## 1. Data Loading and Preprocessing

### 1.1 Reading the Data

# Read the data containing molecular descriptors and Ki values
bace_data_mordred = pd.read_csv('combined_bace_inhibitors_ki_and_rdkit_descriptors.csv')


## 2. Data Normalization and Feature Selection

### 2.1 Scaling and Saving Ki Values
```python
# Extract and save the unscaled 'Ki (nM)' column
bace_unscaled = bace_data_mordred[['Ki (nM)']]
bace_unscaled.to_csv('bace_ki_unscaled.csv', index=False)

# Drop unnecessary columns for feature selection
X = bace_data_mordred.drop(['BindingDB Reactant_set_id', 'Ligand SMILES', 'BindingDB MonomerID', 'BindingDB Ligand Name', 'Target Name', 'Ki (nM)'], axis=1)
```

### 2.2 Variance Thresholding
```python
# Normalize the data using StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_normalized = scaler.fit_transform(X)

# Scale descriptors using Variance Threshold
from sklearn.feature_selection import VarianceThreshold

# Initialize VarianceThreshold with the chosen threshold (e.g., 0.01)
variance_threshold = VarianceThreshold(threshold=0.01)

# Fit and transform the data using variance thresholding
data_selected = variance_threshold.fit_transform(data_normalized)
```

### 2.3 Principal Component Analysis (PCA)
```python
# Apply PCA for dimensionality reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=100)
data_pca = pca.fit_transform(data_normalized)
```

## 3. Model Building and Training

### 3.1 Convolutional Neural Network (CNN)
```python
# Reshape the input for CNN
X_cnn = data_selected.reshape(data_selected.shape[0], data_selected.shape[1], 1)

# Split the data into training and testing sets
X_cnn_train, X_cnn_test, y_train, y_test = train_test_split(data_normalized, bace_ki, test_size=0.2, random_state=119)

# Define and compile the CNN model
model_cnn = Sequential(...)
model_cnn.compile(optimizer='adam', loss='mean_squared_error')

# Train the CNN model
model_cnn.fit(X_cnn_train, y_train, epochs=10, batch_size=32, validation_data=(X_cnn_test, y_test))
```

### 3.2 Recurrent Neural Network (RNN)
```python
# Reshape the input for RNN
X_rnn = data_normalized.reshape(data_normalized.shape[0], 1, data_normalized.shape[1])

# Split the data into training and testing sets
X_rnn_train, X_rnn_test = X_rnn[y_train.index], X_rnn[y_test.index]

# Define and compile the RNN model
model_rnn = Sequential(...)
model_rnn.compile(optimizer='adam', loss='mean_squared_error')

# Train the RNN model
model_rnn.fit(X_rnn_train, y_train, epochs=10, batch_size=32, validation_data=(X_rnn_test, y_test))
```

## 4. Model Evaluation

### 4.1 Evaluate CNN and RNN Performance
```python
# Make predictions using the trained models
y_pred_cnn = model_cnn.predict(X_cnn_test)
y_pred_rnn = model_rnn.predict(X_rnn_test)

# Calculate evaluation metrics
mse_cnn = mean_squared_error(y_test, y_pred_cnn)
mae_cnn = mean_absolute_error(y_test, y_pred_cnn)
r2_cnn = r2_score(y_test, y_pred_cnn)

mse_rnn = mean_squared_error(y_test, y_pred_rnn)
mae_rnn = mean_absolute_error(y_test, y_pred_rnn)
r2_rnn = r2_score(y_test, y_pred_rnn)

# Print the evaluation metrics
print("CNN Model:")
print(f"MSE: {mse_cnn:.2f}")
print(f"MAE: {mae_cnn:.2f}")
print(f"R-squared: {r2_cnn:.2f}")

print("\nRNN Model:")
print(f"MSE: {mse_rnn:.2f}")
print(f"MAE: {mae_rnn:.2f}")
print(f"R-squared: {r2_rnn:.2f}")
```

### 4.2 Evaluate Extra Trees Regressor
```python
# Evaluate Extra Trees Regressor
r2score = r2_score(y_test, y_predict)
mse = mean_squared_error(y_predict, y_test)
rmse = math.sqrt(mse)

print(f"Extra Trees Regressor Performance:")
print(f"R-squared: {r2score:.2f}")
print(f"RMSE: {rmse:.2f}")
```

### 4.3 Evaluate Models with LazyRegressor
```python
from lazypredict.Supervised import LazyRegressor

# Initialize and run LazyRegressor
reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# Print the list of models and their performance metrics
print("LazyRegressor Models:")
print(models)
```

