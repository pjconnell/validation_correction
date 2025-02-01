import numpy as np
import pandas as pd
from validation_correction import validation_correction

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 10000
n_validation = 2000

# True X (binary predictor)
X = np.random.normal(0, 0.5, n_samples)

# Additional predictor Z
Z = np.random.normal(0, 0.25, n_samples)

# Generate outcome Y
true_beta_0 = 1
true_beta_x = 2
true_beta_z = 3

F= 1/(1+np.exp(-(true_beta_0 + true_beta_x * X + true_beta_z * Z)))

Y = np.random.binomial(1,F)

# Misclassified version of Y (U)
# P(U=1|Y=1) = 0.8 (sensitivity)
# P(U=0|Y=0) = 0.7 (specificity)
U = np.zeros_like(X)
for i in range(n_samples):
    if Y[i] == 1:
        U[i] = np.random.binomial(1, 0.9)
    else:
        U[i] = np.random.binomial(1, 0.3)


# Create main dataset
research_data = pd.DataFrame({
    'u': U,
    'x': X,
    'z': Z
})

# Create validation dataset (random subset with both true Y and observed U)
val_indices = np.random.choice(n_samples, n_validation, replace=False)
validation_data = pd.DataFrame({
    'y': Y[val_indices],
    'u': U[val_indices]
})

print("Testing validation_correction package...")
print("\nTrue coefficients:")
print(f"beta_0: {true_beta_0}")
print(f"beta_x: {true_beta_x}")
print(f"beta_z: {true_beta_z}")

# Run naive regression
print("\nRunning naive regression...")
naive_result = validation_correction.logit(
    formula="u ~ x + z",
    data=research_data,
    val_data=None
)
print("\nNaive regression results:")
print(naive_result.summary().tables[1])

# Run corrected regression
print("\nRunning corrected regression...")
result = validation_correction.logit(
    formula="u||y ~ x + z",  # u is the mismeasurement of y
    data=research_data,
    val_data=validation_data,
    bootstrap=True,
    n_boots=100
)
print("\nCorrected regression results:")
print(result)

# Create plots
print("\nGenerating plots...")
coef_plot = validation_correction.plot_coefficients(naive_result, result)

# Plot bootstrap distributions using the bootstrap results from validation_correction
dist_plot = validation_correction.plot_bootstrap_distributions()

print("\nTest complete.") 
