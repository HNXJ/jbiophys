# src/jbiophysic/models/builders/hierarchy.py
import jaxley as jx
from jaxley.synapses import IonotropicSynapse
from .populations import construct_column
from jbiophysic.common.utils.logging import get_logger

logger = get_logger(__name__)

def build_cortical_hierarchy(n_areas: int = 11) -> jx.Network:
    """Inter-areal connectivity across multiple visual areas."""
    logger.info(f"Building cortical hierarchy with {n_areas} areas")
    
    # 1. Collect all cells from all areas
    # construct_column() now returns a list of cells.
    all_cells = []
    for _ in range(n_areas):
        all_cells.extend(construct_column())
    
    # 2. Combine into one macroscopic object
    brain = jx.Network(all_cells)
    
    # Re-apply population and area groups on the flat network
    # Each area has 5 cells: 2 PC, 1 PV, 1 SST, 1 VIP
    cells_per_area = 5
    for i in range(n_areas):
        base = i * cells_per_area
        brain.cell(list(range(base, base + 2))).add_to_group("PC")
        brain.cell(list(range(base + 2, base + 3))).add_to_group("PV")
        brain.cell(list(range(base + 3, base + 4))).add_to_group("SST")
        brain.cell(list(range(base + 4, base + 5))).add_to_group("VIP")
        # Add area-specific groups for inter-areal routing
        area_name = f"Area_{i}"
        brain.cell(list(range(base, base + cells_per_area))).add_to_group(area_name)
    
    # 3. Inter-Areal Connectivity (Point-to-Point mapping to avoid internal broadcasting errors)
    ff_synapse = IonotropicSynapse()
    fb_synapse = IonotropicSynapse()
    
    # Simple linear chain mapping V1 -> higher order
    cells_per_area = 5
    for i in range(n_areas - 1):
        base_i = i * cells_per_area
        base_next = (i + 1) * cells_per_area
        
        # Feedforward: Area i PC -> Area i+1 PC (Connect 2 PCs)
        for j in range(2):
            jx.connect(
                brain.cell(base_i + j).branch(0).comp(0), 
                brain.cell(base_next + j).branch(0).comp(0), 
                ff_synapse
            )
        
        # Feedback: Area i+1 PC -> Area i SST (Connect 1 PC to 1 SST)
        jx.connect(
            brain.cell(base_next + 0).branch(0).comp(0), 
            brain.cell(base_i + 3).branch(0).comp(0), 
            fb_synapse
        )
        
        # Top-down Disinhibition: Higher area -> VIP (Connect 1 PC to 1 VIP)
        jx.connect(
            brain.cell(base_next + 1).branch(0).comp(0), 
            brain.cell(base_i + 4).branch(0).comp(0), 
            fb_synapse
        )
        
    logger.info("Cortical hierarchy assembly complete.")
    return brain

def build_11_area_hierarchy() -> jx.Network:
    """Legacy alias for build_cortical_hierarchy(n_areas=2)."""
    logger.info("Executing legacy alias: build_11_area_hierarchy (REDUCED TO 2 AREAS)")
    return build_cortical_hierarchy(n_areas=2)
