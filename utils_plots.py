
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os



def plot_2d_scatter_tensor(xv, yv, sample_ix, dir, field_vector = None, savefile = None, cmap = 'viridis'):
    """
    Plot 2D plot with field values from tensor format.
    
    Args:
        xv, yv, zv: Meshgrid coordinate arrays of shape [nx, ny, nz]
        sample_ix: Sample index for naming
        dir: Directory to save plot
        field_vector: Tensor of shape [1, 64, 64, 64] or None
        savefile: Optional filename
    """
    
    # Flatten the meshgrid output for plotting
    xv_flat = xv.flatten()
    yv_flat = yv.flatten()
    
    # Handle field_vector tensor
    if field_vector is not None:
        # if isinstance(field_vector, torch.Tensor):
        #     field_vector = field_vector.detach().cpu().numpy()
        
        # Remove batch dimension and flatten: [1, 64, 64, 64] -> [64*64*64]
        if field_vector.ndim == 3 and field_vector.shape[0] == 1:
            field_vector_flat = field_vector[0].flatten()
        elif field_vector.ndim == 2:
            field_vector_flat = field_vector.flatten()
        else:
            raise ValueError(f"Expected field_vector shape [1, 64, 64] or [64, 64], got {field_vector.shape}")
        
        # Ensure same length as coordinate arrays
        if len(field_vector_flat) != len(xv_flat):
            raise ValueError(f"Mismatch: field_vector has {len(field_vector_flat)} elements, "
                           f"coordinates have {len(xv_flat)} elements")
    
    # Plot the points
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    n_pts = field_vector.shape[2]
    
    if field_vector is not None:
        # Reshape field_vector back to 2D grid for imshow
        field_2d = field_vector[0] if field_vector.ndim == 3 else field_vector
        im = ax.imshow(field_2d, extent=[np.min(xv_flat), np.max(xv_flat), np.min(yv_flat), np.max(yv_flat)], 
                      cmap=cmap, origin='lower', aspect='equal')
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=14)
    else:
        # Create a dummy 2D array for visualization when no field_vector is provided
        dummy_field = np.zeros((n_pts, n_pts))
        im = ax.imshow(dummy_field, extent=[np.min(xv_flat), np.max(xv_flat), np.min(yv_flat), np.max(yv_flat)], 
                      cmap=cmap, origin='lower', aspect='equal')
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=14)
    
    # Set axis limits based on your data
    ax.set_xlim([np.min(xv_flat), np.max(xv_flat)])
    ax.set_ylim([np.min(yv_flat), np.max(yv_flat)])
    
    # Set equal aspect ratio for 2D plot
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X-axis', fontsize=14)
    ax.set_ylabel('Y-axis', fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    
    # Add grid for better visualization
    ax.grid(True, alpha=0.3)
    
    if dir:
        if not os.path.exists(dir):
            os.makedirs(dir)
    
    if savefile is not None:
        plt.savefig(f'{dir}/{savefile}_{sample_ix}.pdf', dpi=300, bbox_inches='tight', format='pdf')
    else:
        #plt.savefig(f"{dir}/2d_scatter_plot_tensor_{sample_ix}.png", dpi=300, bbox_inches='tight')
        plt.show()
    plt.close(fig)



def plot_loss_curve(losses, epochs, plot_dir, val_losses=None):
    # Set non-interactive backend
    matplotlib.use('Agg')
    
    plt.figure()
    plt.plot(np.linspace(0, epochs, epochs), losses, color='black', label='Train Loss')
    
    if val_losses is not None:
        plt.plot(np.linspace(0, epochs, epochs), val_losses, color='red', label='Val Loss')
        plt.legend(fontsize=14)
    
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('loss', fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    plt.grid()
    #plt.title('Loss over epochs')
    plt.savefig(f'{plot_dir}/loss_curve.pdf', format='pdf')
    plt.close()

def plot_iterations_curve(nr_iters, epochs, plot_dir):
    # Set non-interactive backend
    matplotlib.use('Agg')
    
    plt.figure()
    plt.plot(np.linspace(0, epochs, epochs), nr_iters, color = 'black')
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('Newton method iterations', fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    plt.grid()
    #plt.title('Number of Newton iterations over epochs')
    plt.savefig(f'{plot_dir}/iters_curve.pdf', format = 'pdf')
    plt.close()


def plot_residuals_comparison(results, plot_dir):
    """
    Plot residuals vs iterations for all samples.
    Original solver residuals in one color family, refined in another.
    Also plots mean residual curves.
    
    Args:
        results: List of result dictionaries containing 'residuals' and 'residuals_refined'
        plot_dir: Directory to save the plot
    """
    matplotlib.use('Agg')
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Colors for original (blue family) and refined (red/orange family)
    original_color = 'steelblue'
    refined_color = 'coral'
    original_mean_color = 'darkblue'
    refined_mean_color = 'darkred'
    
    # Collect all residual curves for computing mean
    all_original_residuals = []
    all_refined_residuals = []
    max_original_iters = 0
    max_refined_iters = 0
    
    # Plot individual curves
    for i, r in enumerate(results):
        residuals = r.get('residuals', [])
        residuals_refined = r.get('residuals_refined', [])
        # Normalise scalars to 1-element lists so len() is always valid.
        if isinstance(residuals, (int, float, np.floating, np.integer)):
            residuals = [residuals]
        if isinstance(residuals_refined, (int, float, np.floating, np.integer)):
            residuals_refined = [residuals_refined]

        if residuals is not None and len(residuals) > 0:
            iters = np.arange(1, len(residuals) + 1)
            label_orig = 'Original (scratch)' if i == 0 else None
            ax.semilogy(iters, residuals, color=original_color, alpha=0.3, 
                       linewidth=0.8, label=label_orig)
            all_original_residuals.append(residuals)
            max_original_iters = max(max_original_iters, len(residuals))
        
        if residuals_refined is not None and len(residuals_refined) > 0:
            iters_ref = np.arange(1, len(residuals_refined) + 1)
            label_ref = 'Refined (from prediction)' if i == 0 else None
            ax.semilogy(iters_ref, residuals_refined, color=refined_color, alpha=0.3, 
                       linewidth=0.8, label=label_ref)
            all_refined_residuals.append(residuals_refined)
            max_refined_iters = max(max_refined_iters, len(residuals_refined))
    
    # Compute and plot mean residual curves
    if len(all_original_residuals) > 0:
        # Pad shorter arrays with NaN for mean computation
        padded_original = np.full((len(all_original_residuals), max_original_iters), np.nan)
        for i, res in enumerate(all_original_residuals):
            padded_original[i, :len(res)] = res
        
        mean_original = np.nanmean(padded_original, axis=0)
        iters_mean_orig = np.arange(1, max_original_iters + 1)
        ax.semilogy(iters_mean_orig, mean_original, color=original_mean_color, 
                   linewidth=2.5, linestyle='-', label='Mean (Original)')
    
    if len(all_refined_residuals) > 0:
        padded_refined = np.full((len(all_refined_residuals), max_refined_iters), np.nan)
        for i, res in enumerate(all_refined_residuals):
            padded_refined[i, :len(res)] = res
        
        mean_refined = np.nanmean(padded_refined, axis=0)
        iters_mean_ref = np.arange(1, max_refined_iters + 1)
        ax.semilogy(iters_mean_ref, mean_refined, color=refined_mean_color, 
                   linewidth=2.5, linestyle='-', label='Mean (Refined)')
    
    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel('Residual (log scale)', fontsize=14)
    ax.set_title('Solver Convergence: Original vs Refined (ML-initialized)')
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc='upper right', fontsize=14)
    ax.grid(True, alpha=0.3, which='both')
    
    # Set reasonable axis limits
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    save_path = os.path.join(plot_dir, 'residuals_comparison.pdf')
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Residuals comparison plot saved to: {save_path}")
