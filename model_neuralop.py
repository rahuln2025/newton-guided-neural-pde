"""
Neural Operator models for PDE solution prediction.
Drop-in replacement for ImagePredictorUNet in model_pdes.py.

Uses the neuraloperator library (https://github.com/neuraloperator/neuraloperator).
Install: pip install neuraloperator

Interface contract (must match model_pdes.ImagePredictorUNet):
    - __init__(self, config=None)  — reads from config['model_config']
    - forward(self, x)             — input [B, in_channels, N, N] → output [B, 1, N, N]
"""

import torch
import torch.nn as nn

from neuralop.models import FNO


class ImagePredictorFNO(nn.Module):
    """
    Fourier Neural Operator (FNO) for 2D PDE solution prediction.
    
    Learns a mapping between function spaces discretized on regular grids
    using spectral (Fourier) convolutions. Resolution-invariant: can train 
    at one resolution and evaluate at another.
    
    The FNO internally appends a 2D positional grid embedding (x, y coordinates)
    to the input channels before lifting. This is controlled by `positional_embedding`.
    
    Input:  [B, in_channels, N, N]  — parameter fields on regular grid
    Output: [B, out_channels, N, N] — predicted PDE solution field
    
    Config keys (under config['model_config']):
        in_channels        (int):   Number of input channels. Default: 4
        out_channels       (int):   Number of output channels. Default: 1
        n_modes            (list):  Fourier modes per spatial dim [mx, my]. Default: [24, 24]
        hidden_channels    (int):   Width of FNO (latent channel dim). Default: 64
        n_layers           (int):   Number of Fourier layers. Default: 4
        
        # Optional FNO architecture parameters
        lifting_channel_ratio   (float): Ratio of lifting channels to hidden_channels. Default: 2
        projection_channel_ratio(float): Ratio of projection channels to hidden_channels. Default: 2
        positional_embedding    (str):   "grid", None. Default: "grid"
        non_linearity           (str):   Activation function name. Default: "gelu"
        norm                    (str):   "ada_in", "group_norm", "instance_norm", None. Default: None
        
        # Channel MLP (applied after each Fourier layer)
        use_channel_mlp     (bool):  Whether to use channel-wise MLP. Default: True
        channel_mlp_dropout (float): Dropout in channel MLP. Default: 0.0
        channel_mlp_expansion(float): Expansion factor for channel MLP. Default: 0.5
        
        # Skip connections
        fno_skip            (str):   Skip type in FNO layers: "linear", "identity", "soft-gating". Default: "linear"
        channel_mlp_skip    (str):   Skip type in channel MLP: "linear", "identity", "soft-gating". Default: "soft-gating"
        
        # Domain padding (important for non-periodic boundary conditions)
        domain_padding      (float or list): Fractional padding per dim. Default: None
        domain_padding_mode (str):   "one-sided" or "symmetric". Default: "one-sided"
        
        # Tensor factorization (for parameter-efficient variants, i.e. TFNO)
        factorization       (str):   None, "tucker", "cp", "tt". Default: None
        rank                (float): Factorization rank (fraction of full). Default: 1.0
        
        # Weight initialization (applied on top of FNO defaults)
        init_method         (str):   "xavier", "kaiming", "normal", "near_zero", "None"/None. Default: None
        init_gain           (float): Gain for weight init. Default: 0.02
    """
    
    def __init__(self, config=None):
        super(ImagePredictorFNO, self).__init__()
        
        if config is None:
            config = {}
        
        model_config = config.get('model_config', {})
        
        # ---- Required parameters ----
        in_channels = model_config.get('in_channels', 4)
        out_channels = model_config.get('out_channels', 1)
        n_modes = model_config.get('n_modes', [24, 24])
        hidden_channels = model_config.get('hidden_channels', 64)
        n_layers = model_config.get('n_layers', 4)
        
        # ---- Lifting / Projection ----
        lifting_channel_ratio = model_config.get('lifting_channel_ratio', 2)
        projection_channel_ratio = model_config.get('projection_channel_ratio', 2)
        
        # ---- Positional embedding ----
        # "grid" appends (x, y) coordinate channels automatically
        positional_embedding = model_config.get('positional_embedding', 'grid')
        
        # ---- Activation ----
        non_linearity_name = model_config.get('activation', 'gelu')
        non_linearity = self._get_activation(non_linearity_name)
        
        # ---- Normalization ----
        norm = model_config.get('norm', None)
        if norm == 'None':
            norm = None
        
        # ---- Channel MLP ----
        use_channel_mlp = model_config.get('use_channel_mlp', True)
        channel_mlp_dropout = model_config.get('channel_mlp_dropout', 0.0)
        channel_mlp_expansion = model_config.get('channel_mlp_expansion', 0.5)
        channel_mlp_skip = model_config.get('channel_mlp_skip', 'soft-gating')
        
        # ---- Skip connection type in Fourier layers ----
        fno_skip = model_config.get('fno_skip', 'linear')
        
        # ---- Domain padding (for non-periodic BCs) ----
        domain_padding = model_config.get('domain_padding', None)
        if domain_padding == 'None':
            domain_padding = None
        domain_padding_mode = model_config.get('domain_padding_mode', 'one-sided')
        
        # ---- Tensor factorization ----
        factorization = model_config.get('factorization', None)
        if factorization == 'None':
            factorization = None
        rank = model_config.get('rank', 1.0)
        
        # ---- Precision ----
        fno_block_precision = model_config.get('fno_block_precision', 'full')
        
        # ---- Build FNO ----
        self.fno = FNO(
            n_modes=tuple(n_modes),
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            lifting_channel_ratio=lifting_channel_ratio,
            projection_channel_ratio=projection_channel_ratio,
            positional_embedding=positional_embedding,
            non_linearity=non_linearity,
            norm=norm,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            channel_mlp_skip=channel_mlp_skip,
            fno_skip=fno_skip,
            domain_padding=domain_padding,
            factorization=factorization,
            rank=rank,
            fno_block_precision=fno_block_precision,
        )
        
        # ---- Optional weight re-initialization ----
        init_method = model_config.get('init_method', None)
        if init_method == 'None':
            init_method = None
        init_gain = model_config.get('init_gain', 0.02)
        if init_method is not None:
            self.initialize_weights(init_method, init_gain)
        
        # ---- Print summary ----
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f'ImagePredictorFNO initialized:')
        print(f'  in_channels={in_channels}, out_channels={out_channels}')
        print(f'  n_modes={n_modes}, hidden_channels={hidden_channels}, n_layers={n_layers}')
        print(f'  positional_embedding={positional_embedding}, activation={non_linearity_name}')
        print(f'  domain_padding={domain_padding}, mode={domain_padding_mode}')
        print(f'  use_channel_mlp={use_channel_mlp}, channel_mlp_dropout={channel_mlp_dropout}')
        print(f'  fno_skip={fno_skip}, channel_mlp_skip={channel_mlp_skip}')
        print(f'  factorization={factorization}, rank={rank}')
        if init_method is not None:
            print(f'  init_method={init_method}, init_gain={init_gain}')
        print(f'  Total parameters: {n_params:,} | Trainable: {n_trainable:,}')
    
    @staticmethod
    def _get_activation(name):
        """Convert activation name string to torch function."""
        import torch.nn.functional as F
        activations = {
            'gelu': F.gelu,
            'relu': F.relu,
            'silu': F.silu,
            'tanh': torch.tanh,
            'leaky_relu': F.leaky_relu,
        }
        if name not in activations:
            raise ValueError(f"Unsupported activation '{name}'. Choose from: {list(activations.keys())}")
        return activations[name]
    
    def initialize_weights(self, method='xavier', gain=0.02):
        """
        Re-initialize weights on top of FNO's defaults.
        Applies to Conv and Linear layers in the lifting/projection blocks.
        SpectralConv weights (complex Fourier coefficients) are left at FNO defaults.
        
        Args:
            method: 'xavier', 'kaiming', 'normal', 'near_zero'
            gain: Scaling factor for initialization
        """
        for name, m in self.fno.named_modules():
            # Skip SpectralConv layers — their weights are complex-valued Fourier 
            # coefficients and should use the library's default initialization
            if 'convs' in name or 'SpectralConv' in type(m).__name__:
                continue
            
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)):
                self._init_weight(m.weight, method, gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, nn.Linear):
                self._init_weight(m.weight, method, gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        print(f"FNO lifting/projection weights re-initialized using {method} "
              f"(gain={gain}). SpectralConv weights kept at library defaults.")
    
    @staticmethod
    def _init_weight(weight, method, gain):
        """Apply initialization to a single weight tensor."""
        if method == 'kaiming':
            nn.init.kaiming_normal_(weight, mode='fan_out', nonlinearity='relu')
        elif method == 'xavier':
            nn.init.xavier_normal_(weight, gain=gain)
        elif method == 'orthogonal':
            nn.init.orthogonal_(weight, gain=gain)
        elif method == 'normal':
            nn.init.normal_(weight, mean=0, std=gain)
        elif method == 'near_zero':
            nn.init.normal_(weight, mean=0, std=gain / 10)
    
    def forward(self, x, output_shape=None):
        """
        Forward pass through the FNO.
        
        Args:
            x: [B, in_channels, N, N] input parameter fields on regular grid.
               Channels correspond to normalized PDE parameters + BC masks
               (same format as ImagePredictorUNet input).
            output_shape: Optional tuple to specify exact output spatial dims.
                          Useful when input has odd spatial dimensions.
                          If None, output spatial dims match input.
        
        Returns:
            [B, out_channels, N, N] predicted PDE solution field.
        """
        return self.fno(x, output_shape=output_shape)


def main():
    """Smoke test: instantiate FNO, run forward pass at two resolutions."""
    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    # Mimic the real config structure from config_adaptiveM_4ch_4096.yaml
    config = {
        'model_config': {
            'in_channels': 4,
            'out_channels': 1,
            'n_modes': [24, 24],
            'hidden_channels': 64,
            'n_layers': 4,
            'activation': 'gelu',
            'domain_padding': 0.078125,
            'domain_padding_mode': 'one-sided',
            'init_method': 'None',
            'init_gain': 0.02,
        }
    }

    model = ImagePredictorFNO(config=config).to(device)

    # --- Test 1: Forward pass at training resolution (64x64) ---
    print('\n--- Test 1: Forward pass at 64x64 ---')
    x_64 = torch.randn(2, 4, 64, 64, device=device)
    t0 = time.time()
    y_64 = model(x_64)
    dt = time.time() - t0
    print(f'Input:  {x_64.shape}')
    print(f'Output: {y_64.shape}  (took {dt:.4f}s)')
    assert y_64.shape == (2, 1, 64, 64), f'FAIL: expected (2,1,64,64), got {y_64.shape}'
    print('PASS')

    # --- Test 2: Resolution invariance — evaluate at 128x128 ---
    print('\n--- Test 2: Resolution invariance at 128x128 ---')
    x_128 = torch.randn(2, 4, 128, 128, device=device)
    t0 = time.time()
    y_128 = model(x_128)
    dt = time.time() - t0
    print(f'Input:  {x_128.shape}')
    print(f'Output: {y_128.shape}  (took {dt:.4f}s)')
    assert y_128.shape == (2, 1, 128, 128), f'FAIL: expected (2,1,128,128), got {y_128.shape}'
    print('PASS')

    # --- Test 3: Gradient flow ---
    print('\n--- Test 3: Gradient flow ---')
    x = torch.randn(2, 4, 64, 64, device=device, requires_grad=False)
    y = model(x)
    loss = y.mean()
    loss.backward()
    grad_norms = {name: p.grad.norm().item() for name, p in model.named_parameters()
                  if p.grad is not None}
    print(f'Parameters with gradients: {len(grad_norms)}')
    print(f'Grad norm range: [{min(grad_norms.values()):.6f}, {max(grad_norms.values()):.6f}]')
    assert len(grad_norms) > 0, 'FAIL: no gradients computed'
    print('PASS')

    # --- Test 4: Batch size 1 (edge case) ---
    print('\n--- Test 4: Batch size 1 ---')
    x_single = torch.randn(1, 4, 64, 64, device=device)
    y_single = model(x_single)
    assert y_single.shape == (1, 1, 64, 64), f'FAIL: expected (1,1,64,64), got {y_single.shape}'
    print(f'Input: {x_single.shape} -> Output: {y_single.shape}')
    print('PASS')

    print('\n=== All tests passed ===')


if __name__ == '__main__':
    main()
