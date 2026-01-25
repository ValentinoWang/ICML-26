import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def draw_cvpr_architecture():
    # Setup the figure
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Styles
    font_main = {'family': 'sans-serif', 'weight': 'bold', 'size': 10}
    font_sub = {'family': 'sans-serif', 'style': 'italic', 'size': 8}
    
    # Color Palette (CVPR friendly)
    c_backbone = '#E3F2FD' # Light Blue
    c_backbone_edge = '#2196F3' # Blue
    c_core = '#E8F5E9'     # Light Green (Correction)
    c_core_edge = '#4CAF50' # Green
    c_aux = '#FFF3E0'      # Light Orange (Weights)
    c_aux_edge = '#FF9800' # Orange
    c_loss = '#FFEBEE'     # Light Red
    c_loss_edge = '#F44336' # Red
    
    # Helper to draw box
    def draw_box(x, y, w, h, color, edge_color, label, sublabel=None, dashed=False):
        style = '--' if dashed else '-'
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                      linewidth=2, edgecolor=edge_color, facecolor=color, linestyle=style)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.15 if sublabel else 0), label, ha='center', va='center', **font_main)
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.15, sublabel, ha='center', va='center', **font_sub, color='#444')
        return (x, y, w, h)

    # Helper to draw arrow
    def draw_arrow(x1, y1, x2, y2, text=None, color='#333'):
        arrow = patches.FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', mutation_scale=15, color=color, linewidth=1.5)
        ax.add_patch(arrow)
        if text:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8)
            ax.text(mid_x, mid_y, text, ha='center', va='center', fontsize=8, bbox=bbox_props, color='#333')

    # --- 1. Input Section ---
    draw_box(0.5, 3.5, 1.0, 1.0, '#FAFAFA', '#9E9E9E', 'Input Set\nCandidates', r'$\mathcal{X}: [B, N, D]$')
    draw_arrow(1.6, 4.0, 2.2, 4.0, text='Flatten')

    # --- 2. Backbone Section (Blue) ---
    # MLP Encoder
    draw_box(2.2, 3.5, 1.5, 1.0, c_backbone, c_backbone_edge, 'MLP Encoder', 'Pointwise\n$D \\to H$')
    draw_arrow(3.8, 4.0, 4.5, 4.0, text='$[B, N, H]$')
    
    # Transformer
    draw_box(4.5, 3.0, 2.0, 2.0, c_backbone, c_backbone_edge, 'Transformer\nEncoder', 'Self-Attn Layers\nFeature Interaction')
    draw_arrow(6.6, 4.0, 7.5, 4.0, text='Contextualized\n$[B, N, H]$')

    # --- Split Point ---
    # Dot at split
    circle = patches.Circle((7.5, 4.0), 0.08, color='black')
    ax.add_patch(circle)

    # --- 3. Auxiliary Branch (Top - Orange) ---
    draw_arrow(7.5, 4.0, 8.5, 6.0, color=c_aux_edge)
    draw_box(8.5, 5.5, 2.0, 1.0, c_aux, c_aux_edge, 'Weight Head', 'MLP $\\to$ Sigmoid', dashed=True)
    draw_arrow(10.6, 6.0, 11.5, 6.0, text='$w: [B, N, 1]$', color=c_aux_edge)
    
    # Weight Output
    draw_box(11.5, 5.5, 1.0, 1.0, '#FFF', c_aux_edge, 'Weights', 'Auxiliary', dashed=True)

    # --- 4. Core Branch (Bottom - Green) ---
    draw_arrow(7.5, 4.0, 8.5, 2.0, color=c_core_edge)
    
    # Pooling
    draw_box(8.5, 1.5, 1.2, 0.8, c_core, c_core_edge, 'Mean Pool', 'Global Agg')
    draw_arrow(9.8, 1.9, 10.2, 1.9, text='$h_{global}$')
    
    # Correction Head
    draw_box(10.2, 1.5, 1.5, 0.8, c_core, c_core_edge, 'Correct Head', 'MLP $\\to \\Delta\\phi$')
    draw_arrow(11.8, 1.9, 12.5, 1.9, text='$\\Delta\\phi$', color=c_core_edge)
    
    # Final Output
    draw_box(12.5, 1.5, 1.2, 0.8, '#E8F5E9', '#2E7D32', 'Output', '$\\theta_{new}$', dashed=False)

    # --- 5. Losses (Right Side - Red) ---
    # Labels for losses
    ax.text(12.0, 7.5, 'Training Objectives', fontsize=12, weight='bold', color='#D32F2F', ha='center')
    
    # Loss Connections
    # Class Loss
    draw_arrow(12.0, 6.5, 12.0, 7.0, color='#D32F2F') 
    ax.text(12.0, 7.1, '$L_{class}$ (BCE)', fontsize=8, color='#D32F2F', ha='center')
    
    # ESS Loss
    draw_arrow(11.5, 6.0, 11.0, 6.5, color='#D32F2F')
    ax.text(11.0, 6.6, '$L_{ess}$', fontsize=8, color='#D32F2F', ha='center')
    
    # Contract Loss
    draw_arrow(13.1, 2.4, 13.1, 3.5, color='#D32F2F')
    ax.text(13.1, 3.6, '$L_{contract}$', fontsize=8, color='#D32F2F', ha='center')

    # Reg Loss
    draw_arrow(11.0, 2.0, 11.0, 2.8, color='#D32F2F')
    ax.text(11.0, 2.9, '$L_{reg} (||\\Delta\\phi||^2)$', fontsize=8, color='#D32F2F', ha='center')

    # Add Titles
    plt.suptitle('Set-Aware Correction Filter Architecture', fontsize=16, weight='bold', y=0.95)
    
    # Legend area
    ax.text(1.0, 0.5, 'Blue: Backbone\nGreen: Main Branch (Inference)\nOrange: Aux Branch (Training Only)\nRed: Loss terms', 
            fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='#ccc'))

    plt.tight_layout()
    out_dir = Path(__file__).resolve().parent / "Results"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "cvpr_architecture.png", dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    draw_cvpr_architecture()
