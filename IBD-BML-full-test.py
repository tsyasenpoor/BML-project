import os
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
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use('Agg')  # For non-interactive environments
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)


###################################
# Configuration Parameters
###################################

train_file_path = r"/home/tsy21001/bml/ArrayExpress-normalized.csv"
metadata_file_path = r"/home/tsy21001/bml/E-MTAB-11349.sdrf.txt"

# Different feature sets
N_values = [500, 1000]   # top N genes by variance
M_values = [250, 500]    # top M genes by L1 logistic regression

# Different dimensions and iterations
d_values = [5, 10, 20] 
num_iters_values = [20, 50, 100]

# Model hyperparameters
a_prime, b_prime = 1.0, 1.0
c_prime, d_prime = 1.0, 1.0
a, c = 1.0, 1.0  # shape parameters for gamma priors
tau, sigma = 1.0, 1.0  # std dev for normal priors on upsilon and gamma
burn_in = 5

# Directories
base_output_dir = "outputs"

###################################
# Data Loading and Preprocessing
###################################

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
# Functions for model updates
###################################

def update_xi_vectorized(xi, theta, a_prime, b_prime, a):
    shape = a + a_prime
    rate = (a_prime / b_prime) + np.sum(theta, axis=1)
    return np.random.gamma(shape, 1.0 / rate, size=xi.shape)

def update_eta_vectorized(eta, beta, c_prime, d_prime, c):
    shape = c + c_prime
    rate = (c_prime / d_prime) + np.sum(beta, axis=1)
    return np.random.gamma(shape, 1.0 / rate, size=eta.shape)

def neg_log_post_upsilon(upsilon_c, c_idx, upsilon, gamma_params, theta, X_aux, y, tau):
    upsilon_full = upsilon.copy()
    upsilon_full[c_idx,:] = upsilon_c
    Z = theta @ upsilon_full.T + X_aux @ gamma_params.T
    pi = softmax(Z, axis=1)
    f = -np.sum(np.log(pi[np.arange(theta.shape[0]), y] + 1e-12))
    f += 0.5 * np.sum(upsilon_c**2) / (tau**2)

    Y_onehot = np.zeros_like(pi)
    Y_onehot[np.arange(theta.shape[0]), y] = 1
    grad = (theta.T @ (pi[:, c_idx] - Y_onehot[:, c_idx])) + (upsilon_c / (tau**2))

    return float(f), np.array(grad, dtype=np.float64)

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

    return float(f), np.array(grad, dtype=np.float64)

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

    return float(f), np.array(grad, dtype=np.float64)

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

    return float(f), np.array(grad, dtype=np.float64)

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

from scipy.optimize import minimize

def update_upsilon_c(upsilon, gamma_params, theta, X_aux, y, c_idx, tau):
    x0 = upsilon[c_idx].copy()
    def objective(u_c):
        return neg_log_post_upsilon(u_c, c_idx, upsilon, gamma_params, theta, X_aux, y, tau)

    res = minimize(fun=objective, x0=x0, method='L-BFGS-B', jac=True)
    u_c_mode = res.x
    H_approx = np.eye(u_c_mode.shape[0])  # Simplify Hessian approximation for upsilon
    Sigma = np.linalg.inv(H_approx)
    upsilon_c_new = np.random.multivariate_normal(u_c_mode, Sigma)
    upsilon[c_idx,:] = upsilon_c_new
    return upsilon

def update_gamma_c(c_idx, upsilon, gamma_params, theta, X_aux, y, sigma):
    x0 = gamma_params[c_idx, :].copy()
    def objective(g_c):
        return neg_log_post_gamma(g_c, c_idx, upsilon, gamma_params, theta, X_aux, y, sigma)

    res = minimize(fun=objective, x0=x0, method='L-BFGS-B', jac=True)
    g_c_mode = res.x
    H_approx = approximate_hessian(objective, g_c_mode)
    Sigma = np.linalg.inv(H_approx)
    gamma_c_new = np.random.multivariate_normal(g_c_mode, Sigma)
    gamma_params[c_idx, :] = gamma_c_new
    return gamma_params

def update_theta_i(i, theta, beta, X, xi, a, upsilon, gamma_params, X_aux, y):
    x0 = theta[i,:].copy()
    def objective(th_i):
        return neg_log_post_theta_i(th_i, i, theta, beta, X, xi, a, upsilon, gamma_params, X_aux, y)

    res = minimize(fun=objective, x0=x0, method='L-BFGS-B', jac=True)
    th_i_mode = res.x
    H_approx = approximate_hessian(objective, th_i_mode)
    Sigma = np.linalg.inv(H_approx)
    th_i_new = np.random.multivariate_normal(th_i_mode, Sigma)
    theta[i,:] = np.clip(th_i_new, 1e-12, None)
    return theta

def update_beta_j(j, theta, beta, X, eta, c, y):
    x0 = beta[j,:].copy()
    def objective(b_j):
        return neg_log_post_beta_j(b_j, j, theta, beta, X, eta, c, y)

    res = minimize(fun=objective, x0=x0, method='L-BFGS-B', jac=True)
    b_j_mode = res.x
    H_approx = approximate_hessian(objective, b_j_mode)
    Sigma = np.linalg.inv(H_approx)
    b_j_new = np.random.multivariate_normal(b_j_mode, Sigma)
    beta[j,:] = np.clip(b_j_new, 1e-12, None)
    return beta


############################################
# Helper functions for plotting and analysis
############################################

def plot_accuracy_iterations(acc_list, output_path):
    plt.figure()
    plt.plot(range(len(acc_list)), acc_list, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Iterations")
    plt.grid(True)
    plt.savefig(os.path.join(output_path, "accuracy_vs_iterations.png"))
    plt.close()

def plot_dim_vs_performance(results_df, output_path):
    # results_df: DataFrame with columns: ['d', 'iterations', 'accuracy']
    # Plot accuracy vs d for a fixed iterations or show multiple lines
    plt.figure()
    for iters in results_df['iterations'].unique():
        subset = results_df[results_df['iterations'] == iters]
        plt.plot(subset['d'], subset['accuracy'], marker='o', label=f"iters={iters}")
    plt.xlabel("Latent Dimension (d)")
    plt.ylabel("Accuracy")
    plt.title("Dimensionality vs. Performance")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_path, "dimensionality_vs_performance.png"))
    plt.close()

def plot_confusion_matrix(cm, output_path, classes=None):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "confusion_matrix.png"))
    plt.close()

def visualize_factors(theta, y, output_path):
    # Use TSNE for factor visualization
    tsne = TSNE(n_components=2, random_state=42)
    theta_2d = tsne.fit_transform(theta)
    plt.figure()
    scatter = plt.scatter(theta_2d[:,0], theta_2d[:,1], c=y, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Disease Class')
    plt.title("t-SNE Visualization of Latent Factors")
    plt.savefig(os.path.join(output_path, "factor_visualization.png"))
    plt.close()

def plot_gene_loading_distributions(beta, output_path):
    # beta is p x d. Let's just plot a histogram of loadings
    plt.figure()
    plt.hist(beta.flatten(), bins=50, alpha=0.7)
    plt.title("Gene Loading Distributions")
    plt.xlabel("Loading Value")
    plt.ylabel("Count")
    plt.grid(True)
    plt.savefig(os.path.join(output_path, "gene_loading_distributions.png"))
    plt.close()

def plot_auxiliary_effects(gamma_params, output_path, aux_feature_names):
    # gamma_params is kappa x p_aux
    plt.figure()
    for c in range(gamma_params.shape[0]):
        plt.bar(range(gamma_params.shape[1]), gamma_params[c], alpha=0.5, label=f"Class {c}")
    plt.xticks(range(len(aux_feature_names)), aux_feature_names, rotation=45)
    plt.title("Auxiliary Variable Effects")
    plt.ylabel("Coefficient Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "auxiliary_effects.png"))
    plt.close()

def top_genes_by_coefficient(beta, upsilon, gamma_params, gene_names, output_path, top_k=20):
    # Identify top genes by magnitude of regression coefficient on latent factors
    # Actually, upsilon and gamma only relate latent factors and aux vars to classes.
    # Beta relates genes to factors. To get top genes, we can look at the magnitude of beta loadings.
    # Another approach: if interested in class separation, we can look at upsilon *or* 
    # the derived importance of genes by summing absolute loadings across factors weighted by upsilon.
    # For simplicity, let's pick genes by their overall loading magnitude sum across factors.

    gene_importance = np.sum(np.abs(beta), axis=1)  # sum of absolute loadings across factors
    top_indices = np.argsort(gene_importance)[-top_k:]
    top_genes = [(gene_names[i], gene_importance[i]) for i in top_indices[::-1]]
    
    with open(os.path.join(output_path, "top_genes.txt"), "w") as f:
        f.write("Top Genes by Loading Magnitude:\n")
        for g, val in top_genes:
            f.write(f"{g}: {val}\n")

    # Optional: bar plot of top genes
    top_genes_names = [x[0] for x in top_genes]
    top_genes_vals = [x[1] for x in top_genes]
    plt.figure()
    plt.barh(top_genes_names[::-1], top_genes_vals[::-1])
    plt.xlabel("Sum of Absolute Loadings")
    plt.title("Top Genes by Loadings")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "top_genes_loadings.png"))
    plt.close()

#############################################
# Main Experiments
#############################################

gene_names_all = train_data_cleaned.columns

results_summary = []  # to store (N, M, d, iters, accuracy)

for N in N_values:
    for M in M_values:
        # Filtering and Feature Selection
        gene_means = X_full.mean(axis=0)
        mean_threshold = 10.0
        high_mean_idx = np.where(gene_means > mean_threshold)[0]
        X_filtered = X_full[:, high_mean_idx]
        gene_filtered_names = gene_names_all[high_mean_idx]

        gene_vars = X_filtered.var(axis=0)
        top_gene_idx = np.argsort(gene_vars)[-N:]
        X_reduced = X_filtered[:, top_gene_idx]
        gene_reduced_names = gene_filtered_names[top_gene_idx]

        # L1 logistic regression feature selection
        clf = LogisticRegression(penalty='l1', solver='saga', multi_class='multinomial',
                                 max_iter=1000, C=0.1, random_state=42)
        clf.fit(X_reduced, y_full)
        importance = np.sum(np.abs(clf.coef_), axis=0)
        top_features = np.argsort(importance)[-M:]
        X_final = X_reduced[:, top_features]
        selected_gene_names = gene_reduced_names[top_features]

        X = X_final
        X_aux = X_aux_full
        y = y_full

        # Create output dir for this feature set
        feature_output_dir = os.path.join(base_output_dir, f"featureN_{N}_M_{M}")
        os.makedirs(feature_output_dir, exist_ok=True)

        # For dimension vs performance plot, store results
        dim_perf_records = []

        for d_val in d_values:
            for num_iters_val in num_iters_values:
                output_dir = os.path.join(feature_output_dir, f"d{d_val}_iters{num_iters_val}")
                os.makedirs(output_dir, exist_ok=True)

                # Initialization
                n, p = X.shape
                xi = np.random.gamma(shape=a_prime, scale=b_prime / a_prime, size=n)
                eta = np.random.gamma(shape=c_prime, scale=d_prime / c_prime, size=p)

                theta = np.random.gamma(shape=a, scale=1.0, size=(n, d_val))
                beta = np.random.gamma(shape=c, scale=1.0, size=(p, d_val))
                upsilon = np.random.normal(0, tau, size=(kappa, d_val))
                gamma_params = np.random.normal(0, sigma, size=(kappa, p_aux))

                acc_list = []
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

                    # Compute accuracy at this iteration
                    Z = theta @ upsilon.T + X_aux @ gamma_params.T
                    pi = softmax(Z, axis=1)
                    y_pred = np.argmax(pi, axis=1)
                    acc = accuracy_score(y, y_pred)
                    acc_list.append(acc)

                # Final evaluation
                Z = theta @ upsilon.T + X_aux @ gamma_params.T
                pi = softmax(Z, axis=1)
                y_pred = np.argmax(pi, axis=1)

                acc = accuracy_score(y, y_pred)
                cm = confusion_matrix(y, y_pred)
                cr = classification_report(y, y_pred)

                with open(os.path.join(output_dir, "results.txt"), "w") as f_out:
                    f_out.write(f"Results for d={d_val}, num_iters={num_iters_val}, N={N}, M={M}\n")
                    f_out.write(f"Accuracy: {acc}\n")
                    f_out.write("Confusion Matrix:\n")
                    f_out.write(str(cm) + "\n")
                    f_out.write("Classification Report:\n")
                    f_out.write(cr + "\n")

                # Plots
                plot_accuracy_iterations(acc_list, output_dir)
                # We will do dimension vs performance plot later after collecting results from all runs
                plot_confusion_matrix(cm, output_dir, classes=["Normal","Crohn's","UC"])
                visualize_factors(theta, y, output_dir)
                plot_gene_loading_distributions(beta, output_dir)
                plot_auxiliary_effects(gamma_params, output_dir, aux_feature_names=["Age","Sex"])

                # Top genes
                top_genes_by_coefficient(beta, upsilon, gamma_params, selected_gene_names, output_dir, top_k=20)

                results_summary.append((N, M, d_val, num_iters_val, acc))
                dim_perf_records.append({'d': d_val, 'iterations': num_iters_val, 'accuracy': acc})

        # After running all d and iterations for this feature set, plot dim vs performance
        dim_perf_df = pd.DataFrame(dim_perf_records)
        plot_dim_vs_performance(dim_perf_df, feature_output_dir)


##############################################
# The code ends here
##############################################

print("All runs completed successfully. Check the outputs directory for results and plots.")
