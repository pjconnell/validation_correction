# validation_correction

A Python package for measurement error correction in regression analysis using validation data.

## Installation

```bash
pip install validation_correction
```

## Usage

The package provides a simple interface for correcting measurement error in both linear and logistic regression using validation data. The correction is implemented using a bootstrap procedure that:
1. Resamples the validation data to estimate misclassification probabilities
2. Applies these probabilities to a bootstrap sample of the research data
3. Repeats this process to obtain valid confidence intervals

### Linear Regression with Mismeasured Predictor

```python
import pandas as pd
from validation_correction import validation_correction

# Load your data
research_data = pd.read_csv("research_data.csv")
validation_data = pd.read_csv("validation_data.csv")

# Run corrected regression with bootstrap
# Format: y ~ w || x + z
# where x is the true variable and w is its mismeasured version
result = validation_correction.ols(
    formula="y ~ w || x + z",
    data=research_data,
    val_data=validation_data,
    bootstrap=True,  # Bootstrap is required for correction
    n_boots=1000    # Number of bootstrap iterations
)

# Run naive regression (no correction)
naive_result = validation_correction.ols(
    formula="y ~ w + z",
    data=research_data,
    val_data=None
)

# Print results with bootstrap confidence intervals
print(result)

# Plot coefficient comparison
validation_correction.plot_coefficients(naive_result, result)

# Plot bootstrap distributions
validation_correction.plot_bootstrap_distributions()
```

### Logistic Regression with Mismeasured Outcome

```python
# Format: u||y ~ x + z
# where y is the true variable and u is its mismeasured version
result = validation_correction.logit(
    formula="u||y ~ x + z",
    data=research_data,
    val_data=validation_data,
    bootstrap=True,
    n_boots=1000
)

print(result)
```

## Formula Specification

The package uses a special formula syntax to specify the relationship between true and mismeasured variables:

1. For mismeasured predictors:
   - Format: `y ~ w || x + z`
   - Where `x` is the true variable and `w` is its mismeasured version
   - Additional covariates (`z`) are measured without error

2. For mismeasured binary outcomes (multinomial outcomes not yet implemented):
   - Format: `u || y ~ x + z`
   - Where `y` is the true outcome and `u` is its mismeasured version
   - Predictors go on the right side of the `~`

3. For naive regression (no correction):
   - Standard formula format: `u ~ w + z`
   - No `||` operator needed
   - Set `val_data=None`

## Visualization

The package provides two types of visualizations:

1. Coefficient Comparison Plot:
   ```python
   validation_correction.plot_coefficients(naive_result, corrected_result)
   ```
   - Shows point estimates and confidence intervals for both naive and corrected models
   - Useful for comparing the magnitude and direction of bias

2. Bootstrap Distribution Plot:
   ```python
   validation_correction.plot_bootstrap_distributions()
   ```
   - Shows the distribution of coefficient estimates from bootstrap samples
   - Includes 95% confidence interval markers
   - Must run regression with `bootstrap=True` first

Bootstrap confidence intervals:
- Uses percentile method (2.5th and 97.5th percentiles)
- Accessible via `result['[0.025']` and `result['0.975]']`
- Number of bootstrap iterations controlled by `n_boots` parameter
- Bootstrap is required for measurement error correction

## Data Requirements

- Main dataset (`data`): Must contain all variables in the formula
- Validation dataset (`val_data`): Must contain both the true and mismeasured versions of the relevant variable
- Both datasets should be pandas DataFrames

## References
- Estimating and Correcting for Misclassification Error in Empirical Textual Research, by Paul Connell and Jonathan H. Choi available at: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4913179

## License

This project is licensed under the MIT License - see the LICENSE file for details.
