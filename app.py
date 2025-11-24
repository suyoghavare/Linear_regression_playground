import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Linear Regression Playground - CS725",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class DataConfig:
    n_samples: int
    n_features: int
    noise_std: float
    random_state: int
    correlation: float = 0.0


@dataclass
class TrainConfig:
    lr_bg: float
    lr_sgd: float
    epochs: int
    run_cf: bool
    run_bg: bool
    run_sgd: bool
    early_stopping: bool = True
    patience: int = 10


@dataclass
class TrainResult:
    theta: np.ndarray
    rmse: float
    runtime: float
    rmse_history: Optional[np.ndarray] = None
    converged: bool = True


def add_bias_column(X: np.ndarray) -> np.ndarray:
    """Add bias column (intercept term) to feature matrix"""
    m = X.shape[0]
    bias = np.ones((m, 1), dtype=X.dtype)
    return np.concatenate([bias, X], axis=1)


@st.cache_data(show_spinner=False, max_entries=5)
def generate_synthetic_data(cfg: DataConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic regression dataset with controllable properties"""
    rng = np.random.RandomState(cfg.random_state)

    # Generate correlated features if specified
    if cfg.correlation > 0 and cfg.n_features > 1:
        # Create covariance matrix with specified correlation
        cov = np.eye(cfg.n_features) * (1 - cfg.correlation) + cfg.correlation
        X = rng.multivariate_normal(
            mean=np.zeros(cfg.n_features), 
            cov=cov, 
            size=cfg.n_samples
        ).astype(np.float32)
    else:
        X = rng.randn(cfg.n_samples, cfg.n_features).astype(np.float32)

    X_with_bias = add_bias_column(X)

    # Generate true parameters with some structure
    true_theta = rng.randn(cfg.n_features + 1, 1).astype(np.float32)
    # Make some parameters zero to simulate feature selection
    if cfg.n_features > 5:
        zero_mask = rng.rand(cfg.n_features + 1) > 0.7
        true_theta[zero_mask] = 0

    noise = (cfg.noise_std * rng.randn(cfg.n_samples, 1)).astype(np.float32)
    y = X_with_bias @ true_theta + noise

    return X_with_bias, y, true_theta


def compute_rmse(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """Compute Root Mean Square Error"""
    preds = X @ theta
    mse = np.mean((preds - y) ** 2)
    return float(np.sqrt(mse))


def closed_form_solution(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Analytical solution using normal equations"""
    X64 = X.astype(np.float64)
    y64 = y.astype(np.float64)
    
    # Regularization for numerical stability
    lambda_reg = 1e-8
    xtx = X64.T @ X64 + lambda_reg * np.eye(X64.shape[1])
    xty = X64.T @ y64
    theta64 = np.linalg.solve(xtx, xty)  # More stable than pinv for well-conditioned problems
    
    return theta64.astype(np.float32)


def can_run_closed_form(n_samples: int, n_features: int) -> Tuple[bool, str]:
    """Check if closed form solution is feasible with explanation"""
    if n_features > 1000:
        return False, "Too many features (D > 1000) - matrix inversion would be too slow"
    if n_samples > 500_000:
        return False, "Too many samples (N > 500,000) - memory requirements too high"
    if n_samples * (n_features ** 2) > 1e9:
        return False, "Problem size too large - would require excessive computation"
    return True, "Feasible"


def batch_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    epochs: int,
    early_stopping: bool = True,
    patience: int = 10
) -> TrainResult:
    """Batch Gradient Descent with optional early stopping"""
    m, n = X.shape
    theta = np.random.randn(n, 1).astype(np.float32)
    history = []
    best_rmse = float('inf')
    patience_counter = 0
    
    t0 = time.perf_counter()
    for epoch in range(epochs):
        preds = X @ theta
        grad = 2.0 / m * X.T @ (preds - y)
        theta -= lr * grad
        
        current_rmse = compute_rmse(X, y, theta)
        history.append(current_rmse)
        
        # Early stopping
        if early_stopping:
            if current_rmse < best_rmse:
                best_rmse = current_rmse
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
    
    t1 = time.perf_counter()
    
    rmse = compute_rmse(X, y, theta)
    converged = patience_counter < patience if early_stopping else True
    
    return TrainResult(
        theta=theta,
        rmse=rmse,
        runtime=t1 - t0,
        rmse_history=np.array(history, dtype=np.float32),
        converged=converged
    )


def stochastic_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    epochs: int,
    early_stopping: bool = True,
    patience: int = 10
) -> TrainResult:
    """Stochastic Gradient Descent with optional early stopping"""
    m, n = X.shape
    theta = np.random.randn(n, 1).astype(np.float32)
    history = []
    best_rmse = float('inf')
    patience_counter = 0
    
    t0 = time.perf_counter()
    for epoch in range(epochs):
        # Learning rate decay
        current_lr = lr / (1 + 0.01 * epoch)
        
        indices = np.random.permutation(m)
        for idx in indices:
            xi = X[idx : idx + 1]
            yi = y[idx : idx + 1]
            grad = 2.0 * xi.T @ (xi @ theta - yi)
            theta -= current_lr * grad
        
        current_rmse = compute_rmse(X, y, theta)
        history.append(current_rmse)
        
        # Early stopping
        if early_stopping:
            if current_rmse < best_rmse:
                best_rmse = current_rmse
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
    
    t1 = time.perf_counter()
    
    rmse = compute_rmse(X, y, theta)
    converged = patience_counter < patience if early_stopping else True
    
    return TrainResult(
        theta=theta,
        rmse=rmse,
        runtime=t1 - t0,
        rmse_history=np.array(history, dtype=np.float32),
        converged=converged
    )


def plot_loss_curves(results: Dict[str, TrainResult]) -> go.Figure:
    """Plot training convergence curves"""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for i, (name, res) in enumerate(results.items()):
        if res.rmse_history is None:
            continue
            
        epochs = np.arange(1, len(res.rmse_history) + 1)
        
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=res.rmse_history,
                mode="lines+markers",
                name=name,
                line=dict(width=3, color=colors[i % len(colors)]),
                marker=dict(size=4),
                hovertemplate="<b>%{text}</b><br>Epoch: %{x}<br>RMSE: %{y:.4f}<extra></extra>",
                text=[name] * len(epochs)
            )
        )
        
        # Add final value annotation
        if len(res.rmse_history) > 0:
            fig.add_annotation(
                x=epochs[-1],
                y=res.rmse_history[-1],
                text=f"{res.rmse_history[-1]:.4f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                ax=50,
                ay=0,
                bgcolor="white",
                bordercolor=colors[i % len(colors)]
            )

    fig.update_layout(
        title="Training Convergence - RMSE vs Epochs",
        xaxis_title="Epoch",
        yaxis_title="RMSE",
        height=500,
        template="plotly_white",
        hovermode="closest",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        font=dict(size=12)
    )
    
    # Add log scale option for y-axis when values vary widely
    if len(results) > 0:
        all_rmse = np.concatenate([r.rmse_history for r in results.values() if r.rmse_history is not None])
        if np.max(all_rmse) / (np.min(all_rmse) + 1e-8) > 100:
            fig.update_yaxis(type="log")
    
    return fig


def plot_coefficients_comparison(
    true_theta: np.ndarray,
    results: Dict[str, TrainResult]
) -> go.Figure:
    """Compare learned coefficients with true parameters"""
    labels = ["Bias"] + [f"Feature {i}" for i in range(1, true_theta.shape[0])]
    
    fig = go.Figure()
    
    # Plot true parameters
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=true_theta.flatten(),
            mode="markers+lines",
            name="True Parameters",
            marker=dict(size=10, symbol="circle", color="black"),
            line=dict(width=3, color="black"),
            hovertemplate="<b>True</b><br>%{x}<br>Value: %{y:.3f}<extra></extra>"
        )
    )
    
    # Plot learned parameters for each method
    colors = px.colors.qualitative.Set1[1:]  # Skip black used for true params
    for i, (name, res) in enumerate(results.items()):
        if res.theta is not None:
            fig.add_trace(
                go.Scatter(
                    x=labels,
                    y=res.theta.flatten(),
                    mode="markers",
                    name=name,
                    marker=dict(
                        size=8,
                        symbol=f"{i+1}" if i < 10 else "circle",
                        color=colors[i % len(colors)]
                    ),
                    hovertemplate=f"<b>{name}</b><br>%{{x}}<br>Value: %{{y:.3f}}<extra></extra>"
                )
            )
    
    fig.update_layout(
        title="Parameter Comparison: True vs Learned Weights",
        xaxis_title="Parameters",
        yaxis_title="Weight Value",
        height=500,
        template="plotly_white",
        showlegend=True,
        xaxis_tickangle=-45
    )
    
    return fig


def plot_feature_importance(theta: np.ndarray, feature_names: list) -> go.Figure:
    """Plot feature importance based on coefficient magnitudes"""
    # Exclude bias term
    weights = np.abs(theta[1:].flatten())
    names = feature_names[1:]  # Exclude bias
    
    # Sort by importance
    sorted_idx = np.argsort(weights)[::-1]
    sorted_weights = weights[sorted_idx]
    sorted_names = [names[i] for i in sorted_idx]
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=sorted_names[:20],  # Show top 20 features
            y=sorted_weights[:20],
            marker_color='lightblue',
            hovertemplate="<b>%{x}</b><br>Importance: %{y:.3f}<extra></extra>"
        )
    )
    
    fig.update_layout(
        title="Top 20 Feature Importances (Absolute Coefficient Values)",
        xaxis_title="Features",
        yaxis_title="Absolute Weight",
        height=400,
        template="plotly_white",
        xaxis_tickangle=-45
    )
    
    return fig


def plot_regression_diagnostic(
    X: np.ndarray, 
    y: np.ndarray, 
    results: Dict[str, TrainResult],
    method: str = "Best"
) -> go.Figure:
    """Diagnostic plots for regression analysis"""
    if method == "Best":
        method = min(results, key=lambda k: results[k].rmse)
    
    theta = results[method].theta
    predictions = X @ theta
    
    fig = go.Figure()
    
    # Actual vs Predicted
    max_val = max(np.max(y), np.max(predictions))
    min_val = min(np.min(y), np.min(predictions))
    
    fig.add_trace(
        go.Scatter(
            x=y.flatten(),
            y=predictions.flatten(),
            mode='markers',
            name=f'{method} Predictions',
            opacity=0.6,
            marker=dict(size=5, color='blue'),
            hovertemplate="Actual: %{x:.3f}<br>Predicted: %{y:.3f}<extra></extra>"
        )
    )
    
    # Perfect prediction line
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash', color='red'),
            hovertemplate=None
        )
    )
    
    fig.update_layout(
        title=f"Actual vs Predicted Values - {method}",
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        height=500,
        template="plotly_white",
        showlegend=True
    )
    
    return fig


def create_performance_dashboard(results: Dict[str, TrainResult]) -> go.Figure:
    """Create a comprehensive performance dashboard"""
    methods = list(results.keys())
    rmses = [results[m].rmse for m in methods]
    runtimes = [results[m].runtime for m in methods]
    
    fig = go.Figure()
    
    # RMSE bars
    fig.add_trace(
        go.Bar(
            x=methods,
            y=rmses,
            name="RMSE",
            marker_color='lightcoral',
            yaxis='y',
            offsetgroup=1,
            hovertemplate="<b>%{x}</b><br>RMSE: %{y:.4f}<extra></extra>"
        )
    )
    
    # Runtime bars (secondary axis)
    fig.add_trace(
        go.Bar(
            x=methods,
            y=runtimes,
            name="Runtime (s)",
            marker_color='lightblue',
            yaxis='y2',
            offsetgroup=2,
            hovertemplate="<b>%{x}</b><br>Runtime: %{y:.3f}s<extra></extra>"
        )
    )
    
    fig.update_layout(
        title="Performance Comparison: RMSE and Runtime",
        xaxis_title="Method",
        template="plotly_white",
        height=400,
        barmode='group',
        yaxis=dict(
            title="RMSE",
            title_font=dict(color="lightcoral"),
            tickfont=dict(color="lightcoral")
        ),
        yaxis2=dict(
            title="Runtime (seconds)",
            title_font=dict(color="lightblue"),
            tickfont=dict(color="lightblue"),
            anchor="x",
            overlaying="y",
            side="right"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig


def build_detailed_interpretation(
    results: Dict[str, TrainResult],
    data_cfg: DataConfig,
    true_theta: np.ndarray
) -> str:
    """Generate detailed interpretation of results"""
    if not results:
        return "No models were trained. Please select at least one method."
    
    lines = []
    lines.append("## Comprehensive Analysis")
    lines.append("")
    
    # Dataset summary
    lines.append("### Dataset Summary")
    lines.append(f"- **Samples (N):** {data_cfg.n_samples:,}")
    lines.append(f"- **Features (D):** {data_cfg.n_features}")
    lines.append(f"- **Noise Level:** {data_cfg.noise_std}")
    lines.append(f"- **Correlation:** {data_cfg.correlation}")
    lines.append("")
    
    # Performance comparison
    lines.append("### Performance Ranking")
    sorted_methods = sorted(results.keys(), key=lambda m: results[m].rmse)
    
    for i, method in enumerate(sorted_methods):
        res = results[method]
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
        lines.append(f"{medal} **{method}**")
        lines.append(f"  - RMSE: {res.rmse:.4f}")
        lines.append(f"  - Runtime: {res.runtime:.3f}s")
        if hasattr(res, 'converged') and not res.converged:
            lines.append(f"  - Note: Did not fully converge")
        lines.append("")
    
    # Method-specific insights
    lines.append("### Method Insights")
    
    if "Closed Form" in results:
        lines.append("**Closed Form (OLS):**")
        lines.append("- Exact solution (no approximation)")
        lines.append("- Guaranteed optimal for linear regression")
        lines.append("- Computationally expensive for large D")
        lines.append("- Memory intensive for large N×D matrices")
        lines.append("")
    
    if "Batch GD" in results:
        bg_res = results["Batch GD"]
        lines.append("**Batch Gradient Descent:**")
        lines.append("- Stable convergence")
        lines.append("- Uses full dataset for each update")
        lines.append("- Can be slow for large N")
        lines.append("- Requires careful learning rate tuning")
        if bg_res.converged:
            lines.append("- Converged successfully")
        else:
            lines.append("- Did not converge - try lower learning rate or more epochs")
        lines.append("")
    
    if "SGD" in results:
        sgd_res = results["SGD"]
        lines.append("**Stochastic Gradient Descent:**")
        lines.append("- Fast updates (single sample)")
        lines.append("- Good for large datasets")
        lines.append("- Noisy convergence")
        lines.append("- Sensitive to learning rate")
        if sgd_res.converged:
            lines.append("- Converged successfully")
        else:
            lines.append("- Did not converge - try lower learning rate")
        lines.append("")
    
    # Recommendations
    lines.append("### Recommendations")
    
    best_method = sorted_methods[0]
    lines.append(f"**Best performing method:** {best_method}")
    
    if data_cfg.n_features > 100:
        lines.append("- For high-dimensional data, consider SGD or regularization")
    
    if data_cfg.n_samples > 10000:
        lines.append("- For large datasets, iterative methods (SGD) are preferred")
    
    if any(not res.converged for res in results.values()):
        lines.append("- Some methods didn't converge: try decreasing learning rates")
    
    return "\n".join(lines)


def sidebar_data_controls() -> DataConfig:
    """Enhanced sidebar with better organization and explanations"""
    st.sidebar.header("Dataset Configuration")
    
    with st.sidebar.expander("Data Settings", expanded=True):
        n_samples = st.slider(
            "Number of samples (N)",
            min_value=100,
            max_value=500000,
            value=10000,
            step=100,
            help="Total number of data points in the dataset"
        )
        
        n_features = st.slider(
            "Number of features (D)",
            min_value=1,
            max_value=200,
            value=10,
            step=1,
            help="Dimensionality of the feature space"
        )
        
        noise_std = st.slider(
            "Noise standard deviation",
            min_value=0.0,
            max_value=2.0,
            value=0.5,
            step=0.1,
            help="Amount of random noise in the target variable"
        )
        
        correlation = st.slider(
            "Feature correlation",
            min_value=0.0,
            max_value=0.9,
            value=0.0,
            step=0.1,
            help="Correlation between features (0 = independent)"
        )
        
        random_state = st.number_input(
            "Random seed",
            min_value=0,
            max_value=999999,
            value=42,
            step=1,
            help="Seed for reproducible results"
        )
    
    return DataConfig(
        n_samples=n_samples,
        n_features=n_features,
        noise_std=noise_std,
        random_state=random_state,
        correlation=correlation
    )


def sidebar_train_controls() -> TrainConfig:
    """Enhanced training controls with better defaults and explanations"""
    st.sidebar.header("Training Configuration")
    
    with st.sidebar.expander("Optimization Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            lr_bg = st.number_input(
                "Batch GD Learning Rate",
                min_value=1e-5,
                max_value=1.0,
                value=0.01,
                step=0.001,
                format="%.4f",
                help="Step size for batch gradient descent"
            )
            
        with col2:
            lr_sgd = st.number_input(
                "SGD Learning Rate",
                min_value=1e-5,
                max_value=1.0,
                value=0.001,
                step=0.001,
                format="%.4f",
                help="Step size for stochastic gradient descent"
            )
        
        epochs = st.slider(
            "Maximum Epochs",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            help="Maximum number of training iterations"
        )
        
        early_stopping = st.checkbox(
            "Early Stopping",
            value=True,
            help="Stop training if validation doesn't improve"
        )
        
        if early_stopping:
            patience = st.slider(
                "Early Stopping Patience",
                min_value=5,
                max_value=50,
                value=10,
                step=5,
                help="Epochs to wait before stopping if no improvement"
            )
        else:
            patience = 10
    
    st.sidebar.header("Methods Selection")
    with st.sidebar.expander("Choose Algorithms", expanded=True):
        run_cf = st.checkbox(
            "Closed Form (OLS)", 
            value=True,
            help="Analytical solution using normal equations"
        )
        run_bg = st.checkbox(
            "Batch Gradient Descent", 
            value=True,
            help="Full dataset per update - stable but slow"
        )
        run_sgd = st.checkbox(
            "Stochastic Gradient Descent", 
            value=True,
            help="Single sample per update - fast but noisy"
        )
    
    return TrainConfig(
        lr_bg=lr_bg,
        lr_sgd=lr_sgd,
        epochs=epochs,
        run_cf=run_cf,
        run_bg=run_bg,
        run_sgd=run_sgd,
        early_stopping=early_stopping,
        patience=patience
    )


def show_welcome_section():
    """Show comprehensive welcome and instructions"""
    st.title("Linear Regression Playground - Group 1")
    st.markdown("### CS725 - Foundations of Machine Learning")
    
    with st.expander("Getting Started Guide", expanded=True):
        st.markdown("""
        **Welcome!** This interactive playground helps you understand different linear regression optimization methods.
        
        ### What You'll Learn:
        - How different optimization algorithms perform
        - Trade-offs between computational cost and accuracy
        - Effects of dataset size and dimensionality
        - Convergence behavior of gradient-based methods
        
        ### How to Use:
        1. **Configure Dataset** in sidebar (size, complexity, noise)
        2. **Choose Algorithms** to compare
        3. **Set Training Parameters** (learning rates, epochs)
        4. **Run Training** and analyze results
        5. **Explore Different Tabs** for detailed insights
        
        ### Key Comparisons:
        - **Closed Form**: Exact solution, computationally expensive
        - **Batch GD**: Stable, uses full dataset each step  
        - **SGD**: Fast, noisy updates, good for large data
        """)
    
    # Quick tips
    st.info("**Pro Tip**: Start with default settings, then experiment with different dataset sizes and learning rates!")


def show_progress_and_metrics(results: Dict[str, TrainResult]):
    """Show training progress and key metrics in a nice layout"""
    st.subheader("Training Results")
    
    # Create metrics in columns
    cols = st.columns(len(results) + 1)
    
    with cols[0]:
        st.metric("Methods Compared", str(len(results)))
    
    for i, (name, res) in enumerate(results.items(), 1):
        with cols[i]:
            st.metric(
                label=name,
                value=f"RMSE: {res.rmse:.4f}",
                delta=f"{res.runtime:.3f}s",
                delta_color="off"
            )


def main():
    """Main application function"""
    show_welcome_section()
    
    # Configuration
    data_cfg = sidebar_data_controls()
    train_cfg = sidebar_train_controls()
    
    # Generate data
    X, y, true_theta = generate_synthetic_data(data_cfg)
    
    # Show dataset info
    with st.expander("Dataset Overview", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Samples", f"{data_cfg.n_samples:,}")
        with col2:
            st.metric("Features", data_cfg.n_features)
        with col3:
            st.metric("Noise Level", f"{data_cfg.noise_std:.1f}")
        
        # Feature correlation warning
        if data_cfg.correlation > 0.5:
            st.warning(f"High feature correlation ({data_cfg.correlation}) may affect model performance")
    
    # Run training
    st.markdown("---")
    st.header("Run Training")
    
    if not (train_cfg.run_cf or train_cfg.run_bg or train_cfg.run_sgd):
        st.error("Please select at least one training method in the sidebar!")
        return
    
    if st.button("Start Training", type="primary", width='stretch'):
        with st.spinner("Training models... This may take a while for large datasets"):
            results = run_training(X, y, train_cfg, data_cfg)
        
        if not results:
            st.error("No results obtained. Check configuration and try again.")
            return
        
        # Show results
        show_progress_and_metrics(results)
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Performance", "Convergence", "Parameters", 
            "Diagnostics", "Analysis"
        ])
        
        with tab1:
            st.plotly_chart(create_performance_dashboard(results), width='stretch')
        
        with tab2:
            # Convergence plots
            iterative_results = {k: v for k, v in results.items() if v.rmse_history is not None}
            if iterative_results:
                st.plotly_chart(plot_loss_curves(iterative_results), width='stretch')
            else:
                st.info("No iterative methods selected. Enable Batch GD or SGD to see convergence plots.")
        
        with tab3:
            # Parameter analysis
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_coefficients_comparison(true_theta, results), width='stretch')
            with col2:
                if data_cfg.n_features > 1:
                    feature_names = ["Bias"] + [f"Feature {i}" for i in range(1, data_cfg.n_features + 1)]
                    best_method = min(results, key=lambda k: results[k].rmse)
                    st.plotly_chart(
                        plot_feature_importance(results[best_method].theta, feature_names), 
                        width='stretch'
                    )
        
        with tab4:
            # Diagnostic plots
            best_method = min(results, key=lambda k: results[k].rmse)
            st.plotly_chart(
                plot_regression_diagnostic(X, y, results, best_method), 
                width='stretch'
            )
        
        with tab5:
            # Detailed analysis
            interpretation = build_detailed_interpretation(results, data_cfg, true_theta)
            st.markdown(interpretation)


def run_training(
    X: np.ndarray,
    y: np.ndarray,
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
) -> Dict[str, TrainResult]:
    """Run selected training methods with progress tracking"""
    results = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_methods = train_cfg.run_cf + train_cfg.run_bg + train_cfg.run_sgd
    completed = 0
    
    def update_progress(message):
        nonlocal completed
        completed += 1
        progress_bar.progress(completed / total_methods)
        status_text.text(f"Status: {message}")
    
    # Closed Form
    if train_cfg.run_cf:
        feasible, reason = can_run_closed_form(data_cfg.n_samples, data_cfg.n_features)
        if feasible:
            update_progress("Running Closed Form solution...")
            t0 = time.perf_counter()
            theta_cf = closed_form_solution(X, y)
            t1 = time.perf_counter()
            rmse_cf = compute_rmse(X, y, theta_cf)
            results["Closed Form"] = TrainResult(
                theta=theta_cf,
                rmse=rmse_cf,
                runtime=t1 - t0,
            )
        else:
            st.warning(f"Closed Form skipped: {reason}")
    
    # Batch Gradient Descent
    if train_cfg.run_bg:
        update_progress("Running Batch Gradient Descent...")
        res_bg = batch_gradient_descent(
            X, y, train_cfg.lr_bg, train_cfg.epochs,
            train_cfg.early_stopping, train_cfg.patience
        )
        results["Batch GD"] = res_bg
    
    # Stochastic Gradient Descent
    if train_cfg.run_sgd:
        update_progress("Running Stochastic Gradient Descent...")
        res_sgd = stochastic_gradient_descent(
            X, y, train_cfg.lr_sgd, train_cfg.epochs,
            train_cfg.early_stopping, train_cfg.patience
        )
        results["SGD"] = res_sgd
    
    progress_bar.empty()
    status_text.text("Training completed!")
    
    return results


if __name__ == "__main__":
    main()