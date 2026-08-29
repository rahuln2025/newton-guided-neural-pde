import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from diffusers import UNet2DModel

class ImagePredictorUNet(nn.Module):
    def __init__(self, config=None):
        super(ImagePredictorUNet, self).__init__()
        
        # Use config if provided, otherwise use defaults
        if config is None:
            config = {}
        
        model_config = config.get('model_config')
            
        # Get UNet configuration parameters from config or use defaults
        sample_size = model_config.get('sample_size')
        in_channels = model_config.get('in_channels')
        out_channels = model_config.get('out_channels')
        layers_per_block = model_config.get('layers_per_block')
        block_out_channels = model_config.get('block_out_channels')
        norm_num_groups = model_config.get('norm_num_groups')
        down_block_types = model_config.get('down_block_types')
        up_block_types = model_config.get('up_block_types')
        attention_head_dim = model_config.get('attention_head_dim')
        act_fn = model_config.get('activation', 'gelu')
        dropout_rate = model_config.get('dropout')

        print(f'I am UNet2D with actiation: {act_fn}')
        
        # Define the UNet2D model
        self.unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            norm_num_groups=norm_num_groups,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            attention_head_dim=attention_head_dim,
            act_fn = act_fn, 
            dropout = dropout_rate,
        )
        
        # Initialize weights using the specified method
        init_method = model_config.get('init_method')
        if init_method == "None":
            init_method = None
        gain = model_config.get('init_gain')
        kaiming_scale = model_config.get('kaiming_scale', 0.1)
        if init_method is not None:
            self.initialize_weights(init_method, gain, kaiming_scale)
        
    def initialize_weights(self, method='xavier', gain=0.02, kaiming_scale=0.1):
        """Initialize the weights of the UNet model using the specified method.
        
        Args:
            method (str): Initialization method. Options: 'kaiming', 'xavier', 'orthogonal', 
                        'normal', 'zeros', 'near_zero'
            gain (float): The gain parameter used for some initialization methods
        """
        for m in self.unet.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if method == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif method == 'xavier':
                    nn.init.xavier_normal_(m.weight, gain=gain)
                elif method == 'orthogonal':
                    nn.init.orthogonal_(m.weight, gain=gain)
                elif method == 'normal':
                    nn.init.normal_(m.weight, mean=0, std=gain)
                elif method == 'zeros':
                    nn.init.zeros_(m.weight)
                elif method == 'near_zero':
                    # Initialize with very small values close to zero
                    nn.init.normal_(m.weight, mean=0, std=gain/10)  # Using a small fraction of gain
                
                if m.bias is not None:
                    if method == 'near_zero':
                        nn.init.normal_(m.bias, mean=0, std=gain/10)
                    else:
                        nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, nn.BatchNorm2d):
                if method == 'near_zero':
                    nn.init.normal_(m.weight, mean=1, std=gain/10)
                    nn.init.normal_(m.bias, mean=0, std=gain/10)
                else:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, nn.Linear):
                if method == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif method == 'xavier':
                    nn.init.xavier_normal_(m.weight, gain=gain)
                elif method == 'orthogonal':
                    nn.init.orthogonal_(m.weight, gain=gain)
                elif method == 'normal':
                    nn.init.normal_(m.weight, mean=0, std=gain)
                elif method == 'zeros':
                    nn.init.zeros_(m.weight)
                elif method == 'near_zero':
                    nn.init.normal_(m.weight, mean=0, std=gain/10)
                
                if method == 'near_zero':
                    nn.init.normal_(m.bias, mean=0, std=gain/10)
                else:
                    nn.init.constant_(m.bias, 0)
        
        print(f"UNet weights initialized using {method} initialization" + 
            (f" with gain={gain}" if method in ['xavier', 'orthogonal', 'normal', 'near_zero'] else ""))

    def forward(self, x):
        device = x.device
        batch_size = x.shape[0]
        
        # x already has shape [batch_size, 9, n_pts, n_pts]

        # Create dummy timesteps for UNet
        timesteps = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Forward pass through UNet
        output = self.unet(x, timesteps).sample

        return output
    

class ImagePredictorMLP(nn.Module):
    def __init__(self, config=None):
        super(ImagePredictorMLP, self).__init__()
        
        # Use config if provided, otherwise use defaults
        if config is None:
            config = {}
        
        model_config = config.get('model_config', {})
        
        # Get MLP configuration parameters
        input_dim = model_config.get('in_channels', 10)  # PDE parameters vector
        n_pts = model_config.get('n_pts', 64)  # Grid resolution
        output_dim = n_pts * n_pts
        hidden_dims = model_config.get('hidden_dims')
        activation = model_config.get('activation')
        dropout = model_config.get('dropout')
        
        # Build the MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2))
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'silu':
                layers.append(nn.SiLU())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        self.n_pts = n_pts
        
        # Initialize weights
        init_method = model_config.get('init_method')
        if init_method == "None":
            init_method = None
        gain = model_config.get('init_gain')
        if init_method is not None:
            self.initialize_weights(init_method, gain)
    
    def initialize_weights(self, method='xavier', gain=0.02):
        """Initialize the weights of the MLP model using the specified method.
        
        Args:
            method (str): Initialization method. Options: 'kaiming', 'xavier', 'orthogonal', 
                        'normal', 'zeros', 'near_zero'
            gain (float): The gain parameter used for some initialization methods
        """
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                if method == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif method == 'xavier':
                    nn.init.xavier_uniform_(m.weight, gain=gain)
                elif method == 'orthogonal':
                    nn.init.orthogonal_(m.weight, gain=gain)
                elif method == 'normal':
                    nn.init.normal_(m.weight, mean=0, std=gain)
                elif method == 'zeros':
                    nn.init.zeros_(m.weight)
                elif method == 'near_zero':
                    nn.init.normal_(m.weight, mean=0, std=gain/10)
                
                if m.bias is not None:
                    if method == 'near_zero':
                        nn.init.normal_(m.bias, mean=0, std=gain/10)
                    else:
                        nn.init.constant_(m.bias, 0)
        
        print(f"MLP weights initialized using {method} initialization" + 
            (f" with gain={gain}" if method in ['xavier', 'orthogonal', 'normal', 'near_zero'] else ""))
    
    def forward(self, x):
        # x has shape [batch_size, 9] - PDE parameters
        batch_size = x.shape[0]
        
        # Forward pass through MLP
        output = self.mlp(x)
        
        # Reshape to [batch_size, 1, n_pts, n_pts]
        output = output.view(batch_size, 1, self.n_pts, self.n_pts)
        
        return output
