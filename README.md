# Mercedes-Benz Test Bench Time Prediction

This machine-learning project is based on the Mercedes-Benz Greener
Manufacturing Kaggle competition.

The goal is to predict how much time a vehicle configuration requires on the
test bench. A more accurate prediction can help Mercedes-Benz improve test
planning and reduce unnecessary waiting time.

## Business problem

Different vehicle configurations require different test-bench times. Long and
unpredictable tests can delay production.

The machine-learning task is therefore:

> Predict the test-bench time `y` from anonymized vehicle configuration data.

## Dataset

The training dataset contains:

- 4,209 observations
- 377 anonymized input features
- `ID`: identifier of a vehicle configuration
- `y`: target variable representing test-bench time

The feature names such as `X0`, `X1` and `X10` were anonymized by Mercedes-Benz.
Their original technical meanings are not available.

I handled the anonymity by analysing the statistical properties of the
features instead of interpreting their technical names:

- data type
- number of unique values
- constant columns
- categorical and numerical features
- relationship with the target
- model-based predictive information

The original competition datasets are not included in this public repository.

## Project workflow

1. Load and inspect the data
2. Perform exploratory data analysis
3. Remove constant features
4. Prepare categorical and numerical features
5. Create preprocessing and modelling pipelines
6. Compare several regression models
7. Tune the best model
8. Analyse prediction errors
9. Train the final model
10. Generate the final predictions

## Model comparison

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Mean Baseline | 12.476 | 10.143 | 0.000 |
| Ridge Regression | 8.314 | 5.693 | 0.556 |
| Random Forest | 9.173 | 5.903 | 0.459 |
| Gradient Boosting | 8.052 | 5.302 | 0.584 |
| **Tuned Gradient Boosting** | **7.950** | **5.289** | **0.594** |

The tuned Gradient Boosting model achieved the best validation result.

- RMSE `7.950`: typical errors are approximately 7.95 time units
- MAE `5.289`: predictions differ from the actual value by approximately
  5.29 time units on average
- R² `0.594`: the model explains approximately 59.4% of the variation in the
  target

## Main limitation

The model performs well for normal test-bench times but often underestimates
unusually long test times.

This is visible in the residual analysis and represents the main opportunity
for future improvement.

## Repository files

- `04_mercedes-test-bench.ipynb`: complete analysis and modelling workflow
- `Mercedes_Test_Bench_ML_Project_Presentation.pptx`: final presentation
- `pyproject.toml`: Python project configuration and dependencies
- `uv.lock`: reproducible dependency versions

## Run the project

Install the required packages:

```bash
uv sync
