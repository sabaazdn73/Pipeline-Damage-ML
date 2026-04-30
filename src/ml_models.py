import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import sys
import os

# -------------------------------------------
# python path setup to import from other modules
# -------------------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_and_prepare_data():
  
    df = pd.read_csv('../data/pipeline_data.csv')
    
    feature_cols = [
        'pressure',     
        'D_outer',      
        't_wall',        
        'hoop_stress',   
        'axial_stress', 
        'von_mises_stress',      
    ]
    
    X = df[feature_cols]
    y = df['damage_factor']  
    
    return X, y, feature_cols


def train_and_evaluate():
    
    X, y, feature_cols = load_and_prepare_data()
    
    # -------------------------------------------
    # Train-test split
    # -------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # -------------------------------------------
    # Data Scaling
    # -------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # -------------------------------------------
    # Lineat Regression
    # -------------------------------------------
    print("\n--- Training Linear Regression ---")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    
    results['Linear Regression'] = {
        'y_pred': y_pred_lr,
        'mse': mean_squared_error(y_test, y_pred_lr),
        'r2': r2_score(y_test, y_pred_lr),
        'model': lr
    }
    print(f"R²: {results['Linear Regression']['r2']:.4f}")
    print(f"MSE: {results['Linear Regression']['mse']:.6f}")
    
    # -------------------------------------------
    # Random Forest
    # -------------------------------------------
    print("\n--- Training Random Forest ---")
    rf = RandomForestRegressor(
        n_estimators=100,  
        max_depth=10,      
        random_state=42
    )
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    
    results['Random Forest'] = {
        'y_pred': y_pred_rf,
        'mse': mean_squared_error(y_test, y_pred_rf),
        'r2': r2_score(y_test, y_pred_rf),
        'model': rf
    }
    print(f"R²: {results['Random Forest']['r2']:.4f}")
    print(f"MSE: {results['Random Forest']['mse']:.6f}")
    
    # -------------------------------------------
    # Gaussian Process
    # -------------------------------------------
    print("\n--- Training Gaussian Process ---")
    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        random_state=42
    )
    
    gp.fit(X_train_scaled[:200], y_train.iloc[:200])
    y_pred_gp, y_std_gp = gp.predict(X_test_scaled, return_std=True)
    
    results['Gaussian Process'] = {
        'y_pred': y_pred_gp,
        'y_std': y_std_gp,       # uncertainty 
        'mse': mean_squared_error(y_test, y_pred_gp),
        'r2': r2_score(y_test, y_pred_gp),
        'model': gp
    }
    print(f"R²: {results['Gaussian Process']['r2']:.4f}")
    print(f"MSE: {results['Gaussian Process']['mse']:.6f}")
    
    return results, y_test, X_test_scaled


def plot_results(results, y_test):
    
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Pipeline Damage Detection — ML Model Comparison', 
                 fontsize=14, fontweight='bold')
    
    models = list(results.keys())
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    
    for i, (model_name, color) in enumerate(zip(models, colors)):
        
        y_pred = results[model_name]['y_pred']
        r2 = results[model_name]['r2']
        mse = results[model_name]['mse']
        
        # -------------------------------------------
        # predicted vs true plot
        # -------------------------------------------
        ax1 = axes[0, i]
        ax1.scatter(y_test, y_pred, alpha=0.5, color=color, s=20)
        ax1.plot([0.5, 1.0], [0.5, 1.0], 'r--', lw=2, label='Perfect')
        ax1.set_xlabel('True Damage Factor')
        ax1.set_ylabel('Predicted Damage Factor')
        ax1.set_title(f'{model_name}\nR²={r2:.3f}, MSE={mse:.4f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # -------------------------------------------
        # Error distribution
        # -------------------------------------------
        ax2 = axes[1, i]
        errors = y_pred - y_test
        ax2.hist(errors, bins=30, color=color, alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', lw=2)
        ax2.set_xlabel('Prediction Error')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Error Distribution\nMean={errors.mean():.4f}')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../data/model_comparison.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to ../data/model_comparison.png")
    plt.show()


def print_summary(results):
    
    print("\n" + "="*50)
    print("MODEL COMPARISON SUMMARY")
    print("="*50)
    print(f"{'Model':<20} {'R²':>10} {'MSE':>15}")
    print("-"*50)
    for model_name, res in results.items():
        print(f"{model_name:<20} {res['r2']:>10.4f} {res['mse']:>15.6f}")
    print("="*50)
    
    # best model based on R²
    best = max(results.items(), key=lambda x: x[1]['r2'])
    print(f"\nBest Model: {best[0]} (R²={best[1]['r2']:.4f})")
    
    # uncertainty 
    if 'Gaussian Process' in results:
        gp_std = results['Gaussian Process']['y_std']
        print(f"\nGaussian Process Uncertainty:")
        print(f"  Mean uncertainty: ±{gp_std.mean():.4f}")
        print(f"  Max uncertainty:  ±{gp_std.max():.4f}")


# -------------------------------------------
# function main 
# -------------------------------------------
if __name__ == "__main__":
    print("Loading data and training models...")
    results, y_test, X_test_scaled = train_and_evaluate()
    print_summary(results)
    plot_results(results, y_test)