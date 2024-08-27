import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

# Define the models
def absolute_value_model(values, ref_amount, beta):
    return beta * (values - ref_amount)

def divisive_norm_model(values, ref_amount, beta):
    ev = np.mean(values)
    if ev == 0:
        ev = 1e-8  # Add a small constant to avoid division by zero
    return beta * (values - ref_amount) / ev

def monte_carlo_model(values, ref_amount, beta):
    ev = np.zeros_like(values)
    ev[0] = np.mean(values)  # Set the initial expected value to the mean of all values
    for i in range(1, len(values)):
        ev[i] = np.mean(values[:i])
        if ev[i] == 0:
            ev[i] = 1e-8  # Add a small constant to avoid division by zero
    return beta * (values - ref_amount) / ev

def rescorla_wagner_model(values, ref_amount, beta, alpha):
    ev = np.zeros_like(values)
    ev[0] = values[0]
    for i in range(1, len(values)):
        ev[i] = ev[i-1] + alpha * (values[i] - ev[i-1])
        if ev[i] == 0:
            ev[i] = 1e-8  # Add a small constant to avoid division by zero
    return beta * (values - ref_amount) / ev

# Simulate data for each model
def simulate_data(model, n_samples, params):
    values = np.random.rand(n_samples) * 10
    ref_amount = np.random.rand(n_samples) * 5
    if model == 'absolute_value':
        beta = params['beta']
        y = absolute_value_model(values, ref_amount, beta)
    elif model == 'divisive_norm':
        beta = params['beta']
        y = divisive_norm_model(values, ref_amount, beta)
    elif model == 'monte_carlo':
        beta = params['beta']
        y = monte_carlo_model(values, ref_amount, beta)
    elif model == 'rescorla_wagner':
        beta, alpha = params['beta'], params['alpha']
        y = rescorla_wagner_model(values, ref_amount, beta, alpha)
    return values, ref_amount, y

# Fit each model to the simulated data
def fit_model(model, values, ref_amount, y):
    if model == 'absolute_value':
        def objective(params):
            beta = params[0]
            y_pred = absolute_value_model(values, ref_amount, beta)
            return np.mean((y - y_pred)**2)
        x0 = [1.0]
    elif model == 'divisive_norm':
        def objective(params):
            beta = params[0]
            y_pred = divisive_norm_model(values, ref_amount, beta)
            return np.mean((y - y_pred)**2)
        x0 = [1.0]
    elif model == 'monte_carlo':
        def objective(params):
            beta = params[0]
            y_pred = monte_carlo_model(values, ref_amount, beta)
            return np.mean((y - y_pred)**2)
        x0 = [1.0]
    elif model == 'rescorla_wagner':
        def objective(params):
            beta, alpha = params
            y_pred = rescorla_wagner_model(values, ref_amount, beta, alpha)
            return np.mean((y - y_pred)**2)
        x0 = [1.0, 0.5]
    
    result = minimize(objective, x0)
    score = result.fun
    if np.isnan(score) or np.isinf(score):
        score = np.inf  # Assign a large score for invalid values
    return score

# Analyze model identifiability
models = ['absolute_value', 'divisive_norm', 'monte_carlo', 'rescorla_wagner']
n_samples = 926
n_simulations = 50

param_ranges = {
    'absolute_value': {'beta': [1, 10]},
    'divisive_norm': {'beta': [1, 10]},
    'monte_carlo': {'beta': [1, 10]},
    'rescorla_wagner': {'beta': [1, 10], 'alpha': [0.1, 0.8]}
}

confusion_matrix = np.zeros((len(models), len(models)))

for _ in range(n_simulations):
    for i, true_model in enumerate(models):
        params = {param: np.random.uniform(*param_ranges[true_model][param]) for param in param_ranges[true_model]}
        values, ref_amount, y = simulate_data(true_model, n_samples, params)
        
        best_model = None
        best_score = np.inf
        for j, model in enumerate(models):
            score = fit_model(model, values, ref_amount, y)
            if score < best_score:
                best_model = model
                best_score = score
        
        if best_model is not None:
            confusion_matrix[i, models.index(best_model)] += 1

# Normalize the confusion matrix
confusion_matrix /= n_simulations

# Print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix)

# Visualize the confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, cmap='Blues', xticklabels=models, yticklabels=models, ax=ax)
ax.set_xlabel('Predicted Model')
ax.set_ylabel('True Model')
ax.set_title('Confusion Matrix')
plt.tight_layout()

# Visualize the recovery rates
fig, ax = plt.subplots(figsize=(8, 6))
recovery_rates = np.diag(confusion_matrix)
bar_positions = np.arange(len(models))
ax.bar(bar_positions, recovery_rates, align='center', alpha=0.7)
ax.set_xticks(bar_positions)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.set_ylabel('Recovery Rate')
ax.set_title('Model Identifiability')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# Add recovery rate labels on top of each bar
for i, rate in enumerate(recovery_rates):
    ax.text(i, rate + 0.02, f"{rate:.2f}", ha='center')

plt.tight_layout()
plt.show()