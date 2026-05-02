# src/jbiophysic/models/simulation/run.py
import jaxley as jx
from typing import Dict, Any, Optional
from jbiophysic.common.types.simulation import SimulationConfig, SimulationResult
from jbiophysic.common.utils.logging import get_logger

logger = get_logger(__name__)

def run_simulation(
    brain: jx.Network, 
    config: SimulationConfig,
    params: Optional[Dict[str, Any]] = None
) -> SimulationResult:
    """
    Simulation runner for cortical models.
    Integration errors are intentionally surfaced to prevent silent failures.
    """
    logger.info(f"Running simulation for T={config.t_max}ms with dt={config.dt}ms")
    
    # Axis 11: Recording mandatory for integrate in 0.13.0
    if brain.recordings.empty:
        brain.record("v")
    
    # We rely on Jaxley's internal error handling.
    # No silent zero-trace fallbacks are implemented here.
    # Axis 11: Jaxley 0.13.0 integrate returns only v_trace by default.
    v_trace = jx.integrate(
        brain,
        t_max=config.t_max,
        delta_t=config.dt
    )
    currents, state = None, None
    logger.info("Integration successful.")
        
    return SimulationResult(
        v_trace=v_trace,
        currents=currents,
        state=state,
        metadata={
            "t_max": config.t_max, 
            "dt": config.dt,
            "seed": config.seed
        }
    )
