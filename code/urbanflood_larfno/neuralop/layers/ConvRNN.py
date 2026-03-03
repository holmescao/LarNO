import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class ModuleWrapperIgnores2ndArg(nn.Module):
    """
    A wrapper for any nn.Module that ignores a second argument in the forward pass.
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, dummy_arg=None):
        """
        Forward pass which ignores the dummy argument.

        Parameters:
        - x: Input tensor.
        - dummy_arg: A dummy argument which is expected but not used in computation.
        """
        assert dummy_arg is not None
        x = self.module(x)
        return x


class CGRU_cell_V2(nn.Module):
    """
    A ConvGRU cell implementation that can be used in an encoder or decoder configuration.
    """

    def __init__(self, use_checkpoint, input_channels, filter_size,
                 num_features):
        """
        Initializes the CGRU cell with specific configurations for the convolutional operations and gating mechanisms.

        Parameters:
        - use_checkpoint: Boolean indicating whether to use gradient checkpointing to save memory.
        - shape: Dimensions (height, width) of the input feature maps.
        - input_channels: Number of channels in the input feature map.
        - filter_size: Size of the convolution kernel.
        - num_features: Number of output features for each convolution operation.
        - module: Specifies the configuration as 'encoder' or 'decoder' to adjust internal connections and channel dimensions.
        """
        super(CGRU_cell_V2, self).__init__()

        self.input_channels = int(input_channels)
        self.filter_size = filter_size
        self.num_features = int(num_features)
        self.padding = (filter_size - 1) // 2

        self.module = "encoder"

        if self.module == "encoder":
            conv_input_channels = self.input_channels + self.num_features
            conv_output_channels = 2 * self.num_features
        elif self.module == "decoder":
            conv_input_channels = self.input_channels + 2*self.num_features
            conv_output_channels = 2 * self.num_features

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                conv_input_channels,
                conv_output_channels,
                self.filter_size,
                1,
                self.padding,
            ),
            nn.GroupNorm(max(1, conv_output_channels // 32), conv_output_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                conv_input_channels,
                self.num_features,
                self.filter_size,
                1,
                self.padding,
            ),
            nn.GroupNorm(max(1, self.num_features // 32), self.num_features),
        )

        self.use_checkpoint = use_checkpoint
        # self.dummy_tensor = torch.ones(1,
        #                                dtype=torch.float32,
        #                                requires_grad=True)
        # self.conv1_module_wrapper = ModuleWrapperIgnores2ndArg(self.conv1)
        # self.conv2_module_wrapper = ModuleWrapperIgnores2ndArg(self.conv2)


    def forward(self, inputs=None, hidden_state=None, seq_len=1):
        htprev = hidden_state
        x = inputs

        combined_1 = torch.cat((x, htprev), 1)

        if self.use_checkpoint:
            # Create a new wrapper and dummy tensor each call (required for gradient checkpointing)
            wrapper1 = ModuleWrapperIgnores2ndArg(self.conv1)
            dummy_tensor1 = torch.ones(1, dtype=torch.float32, requires_grad=True, device=x.device)
            gates = checkpoint(wrapper1, combined_1, dummy_tensor1)
        else:
            gates = self.conv1(combined_1)

        zgate, rgate = torch.split(gates, self.num_features, dim=1)
        z = torch.sigmoid(zgate)
        r = torch.sigmoid(rgate)
        if self.module == "encoder":

            combined_2 = torch.cat((x, r * htprev), 1)
        elif self.module == "decoder":
            etprev, dtprev = torch.split(htprev, self.num_features, dim=1)
            combined_2 = torch.cat((x, etprev, r * dtprev), 1)

        if self.use_checkpoint:
            # Create wrapper and dummy tensor again for the second conv (required for gradient checkpointing)
            wrapper2 = ModuleWrapperIgnores2ndArg(self.conv2)
            dummy_tensor2 = torch.ones(1, dtype=torch.float32, requires_grad=True, device=x.device)
            ht = checkpoint(wrapper2, combined_2, dummy_tensor2)
        else:
            ht = self.conv2(combined_2)

        ht = torch.tanh(ht)

        if self.module == "encoder":
            htnext = (1 - z) * htprev + z * ht
        elif self.module == "decoder":
            htnext = (1 - z) * dtprev + z * ht

        return htnext


class CGRU_cell(nn.Module):
    """
    A ConvGRU cell implementation that can be used in an encoder or decoder configuration.
    """

    def __init__(self, use_checkpoint, input_channels, filter_size,
                 num_features):
        """
        Initializes the CGRU cell with specific configurations for the convolutional operations and gating mechanisms.

        Parameters:
        - use_checkpoint: Boolean indicating whether to use gradient checkpointing to save memory.
        - shape: Dimensions (height, width) of the input feature maps.
        - input_channels: Number of channels in the input feature map.
        - filter_size: Size of the convolution kernel.
        - num_features: Number of output features for each convolution operation.
        - module: Specifies the configuration as 'encoder' or 'decoder' to adjust internal connections and channel dimensions.
        """
        super(CGRU_cell, self).__init__()

        self.input_channels = int(input_channels)
        self.filter_size = filter_size
        self.num_features = int(num_features)
        self.padding = (filter_size - 1) // 2

        self.module = "encoder"

        if self.module == "encoder":
            conv_input_channels = self.input_channels + self.num_features
            conv_output_channels = 2 * self.num_features
        elif self.module == "decoder":
            conv_input_channels = self.input_channels + 2*self.num_features
            conv_output_channels = 2 * self.num_features

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                conv_input_channels,
                conv_output_channels,
                self.filter_size,
                1,
                self.padding,
            ),
            nn.GroupNorm(max(1, conv_output_channels // 32), conv_output_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                conv_input_channels,
                self.num_features,
                self.filter_size,
                1,
                self.padding,
            ),
            nn.GroupNorm(max(1, self.num_features // 32), self.num_features),
        )

        self.use_checkpoint = use_checkpoint
        self.dummy_tensor = torch.ones(1,
                                       dtype=torch.float32,
                                       requires_grad=True)
        self.conv1_module_wrapper = ModuleWrapperIgnores2ndArg(self.conv1)
        self.conv2_module_wrapper = ModuleWrapperIgnores2ndArg(self.conv2)


    def forward(self, inputs=None, hidden_state=None, seq_len=1):
        """
        Forward pass for sequential data processing through ConvGRU cell.

        Parameters:
        - inputs: Input tensor sequence.
        - hidden_state: Initial hidden state.
        - seq_len: Sequence length for processing.
        """

        htprev = hidden_state
        x = inputs

        combined_1 = torch.cat((x, htprev), 1)  # X_t + H_t-1
        if self.use_checkpoint:
            gates = checkpoint(self.conv1_module_wrapper, combined_1,
                                self.dummy_tensor)
        else:
            gates = self.conv1(combined_1)  # W * (X_t + H_t-1)

        zgate, rgate = torch.split(gates, self.num_features, dim=1)
        z = torch.sigmoid(zgate)
        r = torch.sigmoid(rgate)

        # h' = tanh(W*(x+r*H_t-1))
        if self.module == "encoder":
            combined_2 = torch.cat((x, r * htprev), 1)
        elif self.module == "decoder":
            etprev, dtprev = torch.split(htprev, self.num_features, dim=1)
            combined_2 = torch.cat((x, etprev, r * dtprev), 1)

        if self.use_checkpoint:
            ht = checkpoint(self.conv2_module_wrapper, combined_2,
                            self.dummy_tensor)
        else:
            ht = self.conv2(combined_2)  # num_features
        ht = torch.tanh(ht)

        if self.module == "encoder":
            htnext = (1 - z) * htprev + z * ht
        elif self.module == "decoder":
            # decoder_htprev = htprev[:, htprev.shape[1]//2:]
            htnext = (1 - z) * dtprev + z * ht

        # output_inner.append(htnext)
        # htprev = htnext

        # return torch.stack(output_inner)
        return htnext

    

def initialize_states(n_layers, shape, device):
    """
    Initializes or resets the states for the encoder and decoder.

    Parameters:
    - device: The device to which the state tensors are to be moved.

    Returns:
    - Tuple of tensors representing the initialized states.
    """
    prev_hidden_state_list = []
    B, C, H, W = shape
    for _ in range(n_layers):
        prev_hidden_state_list.append(torch.zeros(
            (B, C, H, W)).to(device, dtype=torch.float32))

    return prev_hidden_state_list