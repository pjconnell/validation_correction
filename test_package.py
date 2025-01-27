import numpy as np
import pandas as pd
from validation_correction import validation_correction

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 1000
n_validation = 200

# True X (binary predictor)
X = np.random.binomial(1, 0.3, n_samples)

# Misclassified version of X (W)
# P(W=1|X=1) = 0.8 (sensitivity)
# P(W=0|X=0) = 0.9 (specificity)
W = np.zeros_like(X)
for i in range(n_samples):
    if X[i] == 1:
        W[i] = np.random.binomial(1, 0.8)
    else:
        W[i] = np.random.binomial(1, 0.1)

# Additional predictor Z
Z = np.random.normal(0, 1, n_samples)
T = np.random.normal(0, 1, n_samples)

# Generate outcome Y
true_beta_0 = 1.0
true_beta_x = -2.5
true_beta_z = 0.5
true_beta_t = 0.25

Y = true_beta_0 + true_beta_x * X + true_beta_z * Z + true_beta_t * T + np.random.normal(0, 1, n_samples)

# Create main dataset
research_data = pd.DataFrame({
    'y': Y,
    'w': W,
    'z': Z,
    'text_name': T
})

# Create validation dataset (random subset with both true X and observed W)
val_indices = np.random.choice(n_samples, n_validation, replace=False)
validation_data = pd.DataFrame({
    'x': X[val_indices],
    'w': W[val_indices]
})

print("Testing validation_correction package...")
print("\nTrue coefficients:")
print(f"beta_0: {true_beta_0}")
print(f"beta_x: {true_beta_x}")
print(f"beta_z: {true_beta_z}")
print(f"beta_t: {true_beta_t}")

# Run naive regression
print("\nRunning naive regression...")
naive_result = validation_correction.ols(
    formula="y ~ text_name + w + z ",
    data=research_data,
    val_data=None
)
print("\nNaive regression results:")
print(naive_result.summary().tables[1])

# Run corrected regression
print("\nRunning corrected regression...")
result = validation_correction.ols(
    formula="y ~ text_name + w || x + z",  # w is the mismeasurement of x
    data=research_data,
    val_data=validation_data,
    bootstrap=True,
    n_boots=100
)
print(result)

# Create plots
print("\nGenerating plots...")
coef_plot = validation_correction.plot_coefficients(naive_result, result)
coef_plot.savefig('coefficient_comparison.png')
print("Saved coefficient comparison plot to 'coefficient_comparison.png'")

dist_plot = validation_correction.plot_bootstrap_distributions()
dist_plot.savefig('bootstrap_distributions.png')
print("Saved bootstrap distributions plot to 'bootstrap_distributions.png'")

print("\nTest complete. Check the plots in 'coefficient_comparison.png' and 'bootstrap_distributions.png'.") 