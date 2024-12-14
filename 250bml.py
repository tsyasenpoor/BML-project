import numpy as np
import pandas as pd
from scipy.special import softmax, gammaln, logsumexp
from scipy.optimize import minimize
from scipy.stats import gamma as gamma_dist
from time import time
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

from joblib import Parallel, delayed


np.random.seed(42)

###################################
# Step 1: Load and Preprocess Data
###################################

train_file_path = r"/home/tsy21001/bml/ArrayExpress-normalized.csv"
metadata_file_path = r"/home/tsy21001/bml/E-MTAB-11349.sdrf.txt"

def process_gene_expression_data(file_path, metadata_path):
    matrix = pd.read_csv(file_path)
    matrix = matrix.set_index('gene')
    matrix = matrix.T

    features_df = pd.read_csv(metadata_path, sep='\t', index_col=0)
    return matrix, features_df

train_data, train_features = process_gene_expression_data(train_file_path, metadata_file_path)
train_data_cleaned = train_data.drop(['Unnamed: 0', 'refseq'], errors='ignore')
train_data_cleaned.index.name = 'Source Name'

columns_to_keep = ['Characteristics[age]', 'Characteristics[sex]', 'Characteristics[disease]']
filtered_train_features = train_features[columns_to_keep]
filtered_train_features = filtered_train_features.rename(columns={
    'Characteristics[age]': 'Age',
    'Characteristics[sex]': 'Sex',
    'Characteristics[disease]': 'Disease'
})

disease_mapping = {
    'normal': 0,
    "Crohn's disease": 1,
    'ulcerative colitis': 2
}

filtered_train_features['Disease_Label'] = filtered_train_features['Disease'].map(disease_mapping)
filtered_train_features['Age'] = pd.to_numeric(filtered_train_features['Age'], errors='coerce')

common_samples = train_data_cleaned.index.intersection(filtered_train_features.index)
X_full = train_data_cleaned.loc[common_samples].values  # n x p (full gene set)
aux_vars = filtered_train_features.loc[common_samples, ['Age', 'Sex']]

aux_vars['Sex'] = (aux_vars['Sex'].str.lower() == 'female').astype(int)
X_aux_full = aux_vars[['Age','Sex']].values  # n x p_aux
y_full = filtered_train_features.loc[common_samples,'Disease_Label'].values

n, p_full = X_full.shape
p_aux = X_aux_full.shape[1]
kappa = 3

print("Original Data shapes:", X_full.shape, X_aux_full.shape, y_full.shape)

###################################
# Step 2: Gene Filtering
###################################

# 2.1 Mean expression thresholding
gene_means = X_full.mean(axis=0)
mean_threshold = 10.0  # Example threshold
high_mean_idx = np.where(gene_means > mean_threshold)[0]
X_filtered = X_full[:, high_mean_idx]
print(f"After mean filtering: {X_filtered.shape[1]} genes remain.")

# 2.2 Variance filtering
N = 1000  # choose top N genes by variance
gene_vars = X_filtered.var(axis=0)
top_gene_idx = np.argsort(gene_vars)[-N:]
X_reduced = X_filtered[:, top_gene_idx]
print(f"After variance filtering: {X_reduced.shape[1]} genes remain.")

# Optional: Additional feature selection using L1 logistic regression
use_additional_feature_selection = True
if use_additional_feature_selection:
    clf = LogisticRegression(penalty='l1', solver='saga', multi_class='multinomial', max_iter=1000, C=0.1, random_state=42)
    clf.fit(X_reduced, y_full)
    importance = np.sum(np.abs(clf.coef_), axis=0)
    M = 250
    top_features = np.argsort(importance)[-M:]
    X_final = X_reduced[:, top_features]
    print(f"After L1 logistic regression feature selection: {X_final.shape[1]} genes remain.")
else:
    X_final = X_reduced

X = X_final
X_aux = X_aux_full
y = y_full

n, p = X.shape
print("Final Data shapes after filtering:", X.shape, X_aux.shape, y.shape)

###################################
# Hyperparameters
###################################

a_prime, b_prime = 1.0, 1.0
c_prime, d_prime = 1.0, 1.0
a, c = 1.0, 1.0  # shape parameters for gamma priors
tau, sigma = 1.0, 1.0  # std dev for normal priors on upsilon and gamma
burn_in = 5

###################################
# Define Functions
###################################

# def update_xi(xi, theta, a_prime, b_prime, a):
#     for i in range(n):
#         shape = a + a_prime
#         rate = (a_prime/b_prime) + np.sum(theta[i, :])
#         xi[i] = np.random.gamma(shape, 1.0/rate)
#     return xi

def update_xi_vectorized(xi, theta, a_prime, b_prime, a):
    shape = a + a_prime
    rate = (a_prime / b_prime) + np.sum(theta, axis=1)
    return np.random.gamma(shape, 1.0 / rate, size=xi.shape)

# def update_eta(eta, beta, c_prime, d_prime, c):
#     for j in range(p):
#         shape = c + c_prime
#         rate = (c_prime/d_prime) + np.sum(beta[j,:])
#         eta[j] = np.random.gamma(shape, 1.0/rate)
#     return eta

def update_eta_vectorized(eta, beta, c_prime, d_prime, c):
    shape = c + c_prime
    rate = (c_prime / d_prime) + np.sum(beta, axis=1)
    return np.random.gamma(shape, 1.0 / rate, size=eta.shape)

def neg_log_post_upsilon(upsilon_c, c_idx, upsilon, gamma_params, theta, X_aux, y, tau):
    upsilon_full = upsilon.copy()
    upsilon_full[c_idx,:] = upsilon_c
    Z = theta @ upsilon_full.T + X_aux @ gamma_params.T
    pi = softmax(Z, axis=1)
    f = -np.sum(np.log(pi[np.arange(n), y] + 1e-12))
    f += 0.5 * np.sum(upsilon_c**2) / (tau**2)

    Y_onehot = np.zeros_like(pi)
    Y_onehot[np.arange(n), y] = 1
    grad = (theta.T @ (pi[:, c_idx] - Y_onehot[:, c_idx])) + (upsilon_c / (tau**2))

    f = float(f)
    grad = np.array(grad, dtype=np.float64)
    return f, grad

def neg_log_post_gamma(gamma_c, c_idx, upsilon, gamma_params, theta, X_aux, y, sigma):
    gamma_full = gamma_params.copy()
    gamma_full[c_idx, :] = gamma_c

    Z = theta @ upsilon.T + X_aux @ gamma_full.T
    pi = softmax(Z, axis=1)
    f = -np.sum(np.log(pi[np.arange(len(y)), y] + 1e-12))
    f += 0.5 * np.sum(gamma_c**2) / (sigma**2)

    Y_onehot = np.zeros_like(pi)
    Y_onehot[np.arange(len(y)), y] = 1
    grad = (X_aux.T @ (pi[:, c_idx] - Y_onehot[:, c_idx])) + (gamma_c / (sigma**2))

    f = float(f)
    grad = np.array(grad, dtype=np.float64)
    return f, grad

def neg_log_post_theta_i(theta_i, i, theta, beta, X, xi, a, upsilon, gamma_params, X_aux, y):
    theta_full = theta.copy()
    theta_full[i,:] = theta_i

    rates = theta_i @ beta.T
    x_i = X[i,:]
    rates = np.clip(rates, 1e-12, None)
    ll_poisson = np.sum(x_i * np.log(rates) - rates)

    if np.any(theta_i <= 0):
        return np.inf, np.full_like(theta_i, np.inf)
    ll_prior = np.sum((a-1)*np.log(theta_i) - xi[i]*theta_i)

    Z = theta_full @ upsilon.T + X_aux @ gamma_params.T
    pi = softmax(Z, axis=1)
    ll_class = np.log(pi[i, y[i]] + 1e-12)

    log_post = ll_poisson + ll_prior + ll_class
    f = -log_post

    ratio = (x_i / rates) - 1.0
    grad_poisson = (ratio @ beta)
    grad_prior = ((a-1)/theta_i) - xi[i]

    upsilon_weighted = pi[i,:] @ upsilon
    grad_class = upsilon[y[i],:] - upsilon_weighted
    grad = -(grad_poisson + grad_prior + grad_class)

    f = float(f)
    grad = np.array(grad, dtype=np.float64)
    return f, grad

def neg_log_post_beta_j(beta_j, j, theta, beta, X, eta, c, y):
    beta_full = beta.copy()
    beta_full[j,:] = beta_j
    rates_j = theta @ beta_j
    x_j = X[:, j]
    rates_j = np.clip(rates_j, 1e-12, None)
    ll_poisson = np.sum(x_j * np.log(rates_j) - rates_j)

    if np.any(beta_j <= 0):
        return np.inf, np.full_like(beta_j, np.inf)
    ll_prior = np.sum((c-1)*np.log(beta_j) - eta[j]*beta_j)

    log_post = ll_poisson + ll_prior
    f = -log_post

    ratio = (x_j / rates_j) - 1.0
    grad_poisson = (theta.T @ ratio)
    grad_prior = ((c-1)/beta_j) - eta[j]
    grad = -(grad_poisson + grad_prior)

    f = float(f)
    grad = np.array(grad, dtype=np.float64)
    return f, grad

def approximate_hessian(fun, x, eps=1e-5):
    _, g0 = fun(x)
    n_params = len(x)
    hess = np.zeros((n_params, n_params))
    for i in range(n_params):
        x_eps = x.copy()
        x_eps[i] += eps
        _, g1 = fun(x_eps)
        hess[:, i] = (g1 - g0) / eps
    return 0.5*(hess + hess.T)

def update_upsilon_c(upsilon, gamma_params, theta, X_aux, y, c_idx, tau):
    from scipy.optimize import minimize
    x0 = upsilon[c_idx].copy()
    def objective(u_c):
        f, grad = neg_log_post_upsilon(u_c, c_idx, upsilon, gamma_params, theta, X_aux, y, tau)
        return float(f), np.array(grad, dtype=np.float64)

    res = minimize(fun=objective, x0=x0, method='L-BFGS-B', jac=True)
    u_c_mode = res.x
    H_approx = np.eye(u_c_mode.shape[0])  # Simplify: assume identity for sampling
    Sigma = np.linalg.inv(H_approx)
    upsilon_c_new = np.random.multivariate_normal(u_c_mode, Sigma)
    upsilon[c_idx,:] = upsilon_c_new
    return upsilon

def update_gamma_c(c_idx, upsilon, gamma_params, theta, X_aux, y, sigma):
    from scipy.optimize import minimize
    x0 = gamma_params[c_idx, :].copy()
    def objective(g_c):
        f, grad = neg_log_post_gamma(g_c, c_idx, upsilon, gamma_params, theta, X_aux, y, sigma)
        return float(f), np.array(grad, dtype=np.float64)

    res = minimize(fun=objective, x0=x0, method='L-BFGS-B', jac=True)
    g_c_mode = res.x
    H_approx = approximate_hessian(objective, g_c_mode)
    Sigma = np.linalg.inv(H_approx)
    gamma_c_new = np.random.multivariate_normal(g_c_mode, Sigma)
    gamma_params[c_idx, :] = gamma_c_new
    return gamma_params

def update_theta_i(i, theta, beta, X, xi, a, upsilon, gamma_params, X_aux, y):
    from scipy.optimize import minimize
    x0 = theta[i,:].copy()
    def objective(th_i):
        f, grad = neg_log_post_theta_i(th_i, i, theta, beta, X, xi, a, upsilon, gamma_params, X_aux, y)
        return float(f), np.array(grad, dtype=np.float64)

    res = minimize(fun=objective, x0=x0, method='L-BFGS-B', jac=True)
    th_i_mode = res.x
    H_approx = approximate_hessian(objective, th_i_mode)
    Sigma = np.linalg.inv(H_approx)
    th_i_new = np.random.multivariate_normal(th_i_mode, Sigma)
    theta[i,:] = np.clip(th_i_new, 1e-12, None)
    return theta

def update_beta_j(j, theta, beta, X, eta, c, y):
    from scipy.optimize import minimize
    x0 = beta[j,:].copy()
    def objective(b_j):
        f, grad = neg_log_post_beta_j(b_j, j, theta, beta, X, eta, c, y)
        return float(f), np.array(grad, dtype=np.float64)

    res = minimize(fun=objective, x0=x0, method='L-BFGS-B', jac=True)
    b_j_mode = res.x
    H_approx = approximate_hessian(objective, b_j_mode)
    Sigma = np.linalg.inv(H_approx)
    b_j_new = np.random.multivariate_normal(b_j_mode, Sigma)
    beta[j,:] = np.clip(b_j_new, 1e-12, None)
    return beta

########################################################
# Main Experiment Loop
########################################################

def run_experiment(d_val, num_iters_val):
    # Initialization
    xi = np.random.gamma(shape=a_prime, scale=b_prime / a_prime, size=n)
    eta = np.random.gamma(shape=c_prime, scale=d_prime / c_prime, size=p)

    theta = np.random.gamma(shape=a, scale=1.0, size=(n, d_val))
    beta = np.random.gamma(shape=c, scale=1.0, size=(p, d_val))
    upsilon = np.random.normal(0, tau, size=(kappa, d_val))
    gamma_params = np.random.normal(0, sigma, size=(kappa, p_aux))

    # Gibbs Sampling
    for it in range(num_iters_val):
        xi = update_xi_vectorized(xi, theta, a_prime, b_prime, a)
        eta = update_eta_vectorized(eta, beta, c_prime, d_prime, c)

        for j in range(p):
            beta = update_beta_j(j, theta, beta, X, eta, c, y)
        for i in range(n):
            theta = update_theta_i(i, theta, beta, X, xi, a, upsilon, gamma_params, X_aux, y)
        for c_idx in range(kappa):
            upsilon = update_upsilon_c(upsilon, gamma_params, theta, X_aux, y, c_idx, tau)
            gamma_params = update_gamma_c(c_idx, upsilon, gamma_params, theta, X_aux, y, sigma)

        if it % 10 == 0:
            print(f"[d={d_val}, iters={num_iters_val}] Iteration {it}/{num_iters_val}")

    # Predictions
    Z = theta @ upsilon.T + X_aux @ gamma_params.T
    pi = softmax(Z, axis=1)
    y_pred = np.argmax(pi, axis=1)

    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    cr = classification_report(y, y_pred)

    # Save results
    result_filename = f"results_d={d_val}_iters={num_iters_val}_250.txt"
    with open(result_filename, "w") as f_out:
        f_out.write(f"Results for d={d_val}, num_iters={num_iters_val}\n")
        f_out.write(f"Accuracy: {acc}\n")
        f_out.write("Confusion Matrix:\n")
        f_out.write(str(cm) + "\n")
        f_out.write("Classification Report:\n")
        f_out.write(cr + "\n")

    print(f"Finished d={d_val}, num_iters={num_iters_val}. Results saved to {result_filename}.")

d_values = [10, 20, 50, 100]
num_iters_values = [50, 100, 1000]

Parallel(n_jobs=-1)(
    delayed(run_experiment)(d_val, num_iters_val)
    for d_val in d_values for num_iters_val in num_iters_values
)