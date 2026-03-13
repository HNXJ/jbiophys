# jbiophys: Next-Gen Biophysical Modeling

Production repository for hierarchical cortical simulations, adaptive GSDR optimization, and multi-area signal analysis.

## 🛠 Features
- **Adaptive GSDR (AGSDR v2)**: Variance-smoothed stochastic optimization.
- **Physical Realisticity Barrier**: Dampened ODE updates for float32/Metal stability.
- **Hierarchical Scaling**: Depth-dependent circuit manipulation (V1 -> PFC).

## 📦 Structure
- `gsdr/`: Core modular biophysics library.
- `models/`: Domain-specific circuit definitions.
- `notebooks/`: Research and visualization pipelines.
