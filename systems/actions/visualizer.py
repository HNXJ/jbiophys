import os
import sys
import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from typing import Any, List, Dict, Optional, Tuple
from scipy import signal

# --- Path Setup ---
sys.path.insert(0, '/Users/hamednejat/workspace/Repositories/AAE')
sys.path.insert(0, '/Users/hamednejat/workspace/Repositories/jbiophys')

from systems.visualizers.plot_full_simulation_summary import plot_full_simulation_summary
from systems.visualizers.spatial_3d import plot_network_3d
from systems.visualizers.lfp_tools import estimate_lfp
from systems.visualizers.calculate_firing_rates import calculate_firing_rates

def generate_plotly_summary(voltages, time_axis, dt, output_path, spike_threshold=-20.0):
    """Generates a Plotly-based HTML summary (Raster + Spectrogram)."""
    print(f"   📊 Generating Plotly Dynamics Summary: {output_path}")
    
    num_neurons = voltages.shape[0]
    
    # 1. Raster Data
    spike_indices = []
    spike_times = []
    for i in range(num_neurons):
        spikes = (voltages[i, :-1] < spike_threshold) & (voltages[i, 1:] >= spike_threshold)
        times = time_axis[1:][spikes]
        spike_times.extend(times.tolist())
        spike_indices.extend([i] * len(times))

    # 2. Spectrogram Data
    fs = 1000.0 / dt
    mean_v = np.mean(voltages, axis=0)
    freqs, times, Sxx = signal.spectrogram(mean_v, fs=fs, nperseg=int(250/dt))
    
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Raster Plot", "Mean Spectrogram"), vertical_spacing=0.1)
    
    # Raster Trace
    fig.add_trace(go.Scatter(x=spike_times, y=spike_indices, mode='markers', 
                             marker=dict(size=3, color='white'), name="Spikes"), row=1, col=1)
    
    # Spectrogram Trace
    fig.add_trace(go.Heatmap(x=times*1000, y=freqs, z=np.log10(Sxx + 1e-9), 
                             colorscale='Jet', name="Power"), row=2, col=1)
    fig.update_yaxes(title_text="Neuron Index", row=1, col=1)
    fig.update_yaxes(title_text="Frequency (Hz)", range=[1, 100], row=2, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=2, col=1)
    
    fig.update_layout(height=1000, width=1000, template="plotly_dark", title_text="Network Dynamics Summary")
    fig.write_html(output_path)

def run_visualizer_pipeline(
    net: jx.Network,
    params: Any,
    meta: List[Dict],
    output_dir: str = "/Users/hamednejat/workspace/Computational/mscz/figures",
    t_max: float = 1500.0,
    dt: float = 0.1,
    save_svg: bool = False
):
    """Generates the full 8-10 figure biophysical report, saving figures as HTML."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"🎬 Visualizer Pipeline: Generating reports in {output_dir}...")

    # 1. Simulation
    net.cell('all').branch(0).loc(0.0).record('v')
    print(f"   🚀 Simulating {t_max}ms for analysis...")
    voltages = jx.integrate(net, params=params, delta_t=dt, t_max=t_max)
    time_axis = np.arange(voltages.shape[1]) * dt

    # 2. Standard Dynamics Report (Plotly HTML)
    summary_html = os.path.join(output_dir, "simulation_summary.html")
    generate_plotly_summary(voltages, time_axis, dt, summary_html)

    # 2b. Optional SVG (Matplotlib)
    if save_svg:
        print("   📊 Generating Static SVG Summary...")
        summary_svg = os.path.join(output_dir, "simulation_summary.svg")
        plot_full_simulation_summary(voltages, time_axis, dt, save=True, savename=summary_svg)

    # 3. 3D Architecture (Figure 5)
    print("   🌐 Generating 3D Architecture (HTML)...")
    report_3d = os.path.join(output_dir, "network_3d.html")
    plot_network_3d(net, meta, report_3d)

    # 4. Auxiliary Biophysics (Figures 6-10)
    print("   📈 Generating Biophysical Analysis Suite (HTML)...")
    fig_extra = make_subplots(
        rows=5, cols=1,
        subplot_titles=("Population Average Vm", "Biophysical LFP Proxy", "Firing Rate Distribution", "gAMPA Weights", "gGABAa Weights"),
        vertical_spacing=0.05
    )

    # 4.1 Avg Vm
    fig_extra.add_trace(go.Scatter(x=time_axis, y=np.mean(voltages, axis=0), name="Avg Vm", line=dict(color='gold')), row=1, col=1)

    # 4.2 LFP
    lfp = estimate_lfp(voltages, meta, dt)
    fig_extra.add_trace(go.Scatter(x=time_axis, y=lfp, name="LFP", line=dict(color='magenta')), row=2, col=1)

    # 4.3 FR Dist
    frs = calculate_firing_rates(voltages, dt).flatten()
    fig_extra.add_trace(go.Histogram(x=frs, name="FRs", marker_color='cyan'), row=3, col=1)

    # 4.4/4.5 Weight Dists
    # Handle list of dicts or nested dicts safely
    try:
        g_ampa = params[0].get('gAMPA', np.zeros(1))
        g_gaba = params[1].get('gGABAa', np.zeros(1))
    except (KeyError, IndexError):
        g_ampa, g_gaba = np.zeros(1), np.zeros(1)
        
    fig_extra.add_trace(go.Histogram(x=g_ampa, name="gAMPA", marker_color='orange'), row=4, col=1)
    fig_extra.add_trace(go.Histogram(x=g_gaba, name="gGABAa", marker_color='blue'), row=5, col=1)

    fig_extra.update_layout(height=1500, width=1000, title_text="Biophysical Analysis Suite", template="plotly_dark")
    extra_path = os.path.join(output_dir, "biophysical_suite.html")
    fig_extra.write_html(extra_path)

    print(f"✅ Visualizer Pipeline Complete. All figures saved in {output_dir}")
    return voltages
