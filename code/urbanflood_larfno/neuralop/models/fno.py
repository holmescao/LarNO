from torch.utils.checkpoint import checkpoint
from .base_model import BaseModel
from ..layers.complex import ComplexValued
from ..layers.channel_mlp import ChannelMLP
from ..layers.fno_block import FNOBlocks
from ..layers.padding import DomainPadding
from ..layers.spectral_convolution import SpectralConv
from ..layers.embeddings import GridEmbeddingND, GridEmbedding2D
import torch.nn.functional as F
import torch.nn as nn
import torch
from functools import partialmethod
from typing import Tuple, List, Union
from ..layers.ConvRNN import CGRU_cell, initialize_states

import time

Number = Union[float, int]



class ModuleWrapperIgnores2ndArg_gru(nn.Module):
    """
    A module wrapper for a GRU that ignores an additional dummy argument during the forward pass.
    This allows for the use of gradient checkpointing with modules that expect multiple inputs.
    """

    def __init__(self, module):
        """
        Initializes the ModuleWrapperIgnores2ndArg_gru with the specified module.

        Parameters:
        - module: The module to wrap, which should be a GRU or similar recurrent unit.
        """
        super().__init__()
        self.module = module

    def forward(self, x, hx, dummy_arg=None):
        """
        Forwards the inputs through the module while ignoring the dummy argument.

        Parameters:
        - x: The input tensor to the GRU.
        - hx: The hidden state tensor for the GRU.
        - dummy_arg: A dummy argument that is not used but required for API compatibility. Must not be None.

        Returns:
        - Tensor: The output from the GRU module.
        """
        assert dummy_arg is not None, "dummy_arg is required but was None"
        x = self.module(x, hx)
        return x
    
class FNO(BaseModel, name='FNO'):
    """N-Dimensional Fourier Neural Operator. The FNO learns a mapping between
    spaces of functions discretized over regular grids using Fourier convolutions, 
    as described in [1]_.

    The key component of an FNO is its SpectralConv layer (see 
    ``neuralop.layers.spectral_convolution``), which is similar to a standard CNN 
    conv layer but operates in the frequency domain.

    For a deeper dive into the FNO architecture, refer to :ref:`fno_intro`.

    Parameters
    ----------
    n_modes : Tuple[int]
        number of modes to keep in Fourier Layer, along each dimension
        The dimensionality of the FNO is inferred from ``len(n_modes)``
    in_channels : int
        Number of channels in input function
    out_channels : int
        Number of channels in output function
    hidden_channels : int
        width of the FNO (i.e. number of channels), by default 256
    n_layers : int, optional
        Number of Fourier Layers, by default 4

    Documentation for more advanced parameters is below.

    Other parameters
    ------------------
    lifting_channel_ratio : int, optional
        ratio of lifting channels to hidden_channels, by default 2
        The number of liting channels in the lifting block of the FNO is
        lifting_channel_ratio * hidden_channels (e.g. default 512)
    projection_channel_ratio : int, optional
        ratio of projection channels to hidden_channels, by default 2
        The number of projection channels in the projection block of the FNO is
        projection_channel_ratio * hidden_channels (e.g. default 512)
    positional_embedding : Union[str, nn.Module], optional
        Positional embedding to apply to last channels of raw input
        before being passed through the FNO. Defaults to "grid"

        * If "grid", appends a grid positional embedding with default settings to 
        the last channels of raw input. Assumes the inputs are discretized
        over a grid with entry [0,0,...] at the origin and side lengths of 1.

        * If an initialized GridEmbedding module, uses this module directly
        See :mod:`neuralop.embeddings.GridEmbeddingND` for details.

        * If None, does nothing

    non_linearity : nn.Module, optional
        Non-Linear activation function module to use, by default F.gelu
    norm : str {"ada_in", "group_norm", "instance_norm"}, optional
        Normalization layer to use, by default None
    complex_data : bool, optional
        Whether data is complex-valued (default False)
        if True, initializes complex-valued modules.
    channel_mlp_dropout : float, optional
        dropout parameter for ChannelMLP in FNO Block, by default 0
    channel_mlp_expansion : float, optional
        expansion parameter for ChannelMLP in FNO Block, by default 0.5
    channel_mlp_skip : str {'linear', 'identity', 'soft-gating'}, optional
        Type of skip connection to use in channel-mixing mlp, by default 'soft-gating'
    fno_skip : str {'linear', 'identity', 'soft-gating'}, optional
        Type of skip connection to use in FNO layers, by default 'linear'
    resolution_scaling_factor : Union[Number, List[Number]], optional
        layer-wise factor by which to scale the domain resolution of function, by default None

        * If a single number n, scales resolution by n at each layer

        * if a list of numbers [n_0, n_1,...] scales layer i's resolution by n_i.
    domain_padding : Union[Number, List[Number]], optional
        If not None, percentage of padding to use, by default None
        To vary the percentage of padding used along each input dimension,
        pass in a list of percentages e.g. [p1, p2, ..., pN] such that
        p1 corresponds to the percentage of padding along dim 1, etc.
    domain_padding_mode : str {'symmetric', 'one-sided'}, optional
        How to perform domain padding, by default 'one-sided'
    fno_block_precision : str {'full', 'half', 'mixed'}, optional
        precision mode in which to perform spectral convolution, by default "full"
    stabilizer : str {'tanh'} | None, optional
        whether to use a tanh stabilizer in FNO block, by default None

        Note: stabilizer greatly improves performance in the case
        `fno_block_precision='mixed'`. 

    max_n_modes : Tuple[int] | None, optional

        * If not None, this allows to incrementally increase the number of
        modes in Fourier domain during training. Has to verify n <= N
        for (n, m) in zip(max_n_modes, n_modes).

        * If None, all the n_modes are used.

        This can be updated dynamically during training.
    factorization : str, optional
        Tensor factorization of the FNO layer weights to use, by default None.

        * If None, a dense tensor parametrizes the Spectral convolutions

        * Otherwise, the specified tensor factorization is used.
    rank : float, optional
        tensor rank to use in above factorization, by default 1.0
    fixed_rank_modes : bool, optional
        Modes to not factorize, by default False
    implementation : str {'factorized', 'reconstructed'}, optional

        * If 'factorized', implements tensor contraction with the individual factors of the decomposition 

        * If 'reconstructed', implements with the reconstructed full tensorized weight.
    decomposition_kwargs : dict, optional
        extra kwargs for tensor decomposition (see `tltorch.FactorizedTensor`), by default dict()
    separable : bool, optional (**DEACTIVATED**)
        if True, use a depthwise separable spectral convolution, by default False   
    preactivation : bool, optional (**DEACTIVATED**)
        whether to compute FNO forward pass with resnet-style preactivation, by default False
    conv_module : nn.Module, optional
        module to use for FNOBlock's convolutions, by default SpectralConv

    Examples
    ---------

    >>> from neuralop.models import FNO
    >>> model = FNO(n_modes=(12,12), in_channels=1, out_channels=1, hidden_channels=64)
    >>> model
    FNO(
    (positional_embedding): GridEmbeddingND()
    (fno_blocks): FNOBlocks(
        (convs): SpectralConv(
        (weight): ModuleList(
            (0-3): 4 x DenseTensor(shape=torch.Size([64, 64, 12, 7]), rank=None)
        )
        )
            ... torch.nn.Module printout truncated ...

    References
    -----------
    .. [1] :

    Li, Z. et al. "Fourier Neural Operator for Parametric Partial Differential 
        Equations" (2021). ICLR 2021, https://arxiv.org/pdf/2010.08895.

    """

    def __init__(
        self,
        n_modes: Tuple[int],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int = 4,
        lifting_channel_ratio: int = 2,
        projection_channel_ratio: int = 2,
        positional_embedding: Union[str, nn.Module] = "grid",
        non_linearity: nn.Module = F.gelu,
        norm: str = None,
        complex_data: bool = False,
        channel_mlp_dropout: float = 0,
        channel_mlp_expansion: float = 0.5,
        channel_mlp_skip: str = "soft-gating",
        fno_skip: str = "linear",
        resolution_scaling_factor: Union[Number, List[Number]] = None,
        domain_padding: Union[Number, List[Number]] = None,
        domain_padding_mode: str = "one-sided",
        fno_block_precision: str = "full",
        stabilizer: str = None,
        max_n_modes: Tuple[int] = None,
        factorization: str = None,
        rank: float = 1.0,
        fixed_rank_modes: bool = False,
        implementation: str = "factorized",
        decomposition_kwargs: dict = dict(),
        separable: bool = False,
        preactivation: bool = False,
        conv_module: nn.Module = SpectralConv,
        use_checkpoint=True,
        finetune_lifting_only=False,
        **kwargs
    ):

        super().__init__()
        self.n_dim = len(n_modes)

        # n_modes is a special property - see the class' property for underlying mechanism
        # When updated, change should be reflected in fno blocks
        self._n_modes = n_modes

        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers

        # init lifting and projection channels using ratios w.r.t hidden channels
        self.lifting_channel_ratio = lifting_channel_ratio
        self.lifting_channels = lifting_channel_ratio * self.hidden_channels

        self.projection_channel_ratio = projection_channel_ratio
        self.projection_channels = projection_channel_ratio * self.hidden_channels

        self.non_linearity = non_linearity
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = (fno_skip,)
        self.channel_mlp_skip = (channel_mlp_skip,)
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation
        self.complex_data = complex_data
        self.fno_block_precision = fno_block_precision

        if positional_embedding == "grid":
            spatial_grid_boundaries = [[0., 1.]] * self.n_dim
            self.positional_embedding = GridEmbeddingND(in_channels=self.in_channels,
                                                        dim=self.n_dim,
                                                        grid_boundaries=spatial_grid_boundaries)
        elif isinstance(positional_embedding, GridEmbedding2D):
            if self.n_dim == 2:
                self.positional_embedding = positional_embedding
            else:
                raise ValueError(
                    f'Error: expected {self.n_dim}-d positional embeddings, got {positional_embedding}')
        elif isinstance(positional_embedding, GridEmbeddingND):
            self.positional_embedding = positional_embedding
        elif positional_embedding == None:
            self.positional_embedding = None
        else:
            raise ValueError(f"Error: tried to instantiate FNO positional embedding with {positional_embedding},\
                              expected one of \'grid\', GridEmbeddingND")

        if domain_padding is not None and (
            (isinstance(domain_padding, list) and sum(domain_padding) > 0)
            or (isinstance(domain_padding, (float, int)) and domain_padding > 0)
        ):
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                padding_mode=domain_padding_mode,
                resolution_scaling_factor=resolution_scaling_factor,
            )
        else:
            self.domain_padding = None

        self.domain_padding_mode = domain_padding_mode
        self.complex_data = self.complex_data

        if resolution_scaling_factor is not None:
            if isinstance(resolution_scaling_factor, (float, int)):
                resolution_scaling_factor = [
                    resolution_scaling_factor] * self.n_layers
        self.resolution_scaling_factor = resolution_scaling_factor

        self.fno_blocks = FNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=self.n_modes,
            resolution_scaling_factor=resolution_scaling_factor,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            norm=norm,
            preactivation=preactivation,
            fno_skip=fno_skip,
            channel_mlp_skip=channel_mlp_skip,
            complex_data=complex_data,
            max_n_modes=max_n_modes,
            fno_block_precision=fno_block_precision,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            conv_module=conv_module,
            n_layers=n_layers,
            **kwargs
        )

        # if adding a positional embedding, add those channels to lifting
        lifting_in_channels = self.in_channels
        if self.positional_embedding is not None:
            lifting_in_channels += self.n_dim
        # if lifting_channels is passed, make lifting a Channel-Mixing MLP
        # with a hidden layer of size lifting_channels
        if self.lifting_channels:
            self.lifting = ChannelMLP(
                in_channels=lifting_in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.lifting_channels,
                n_layers=2,
                n_dim=self.n_dim,
                non_linearity=non_linearity
            )
        # otherwise, make it a linear layer
        else:
            self.lifting = ChannelMLP(
                in_channels=lifting_in_channels,
                hidden_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                n_layers=1,
                n_dim=self.n_dim,
                non_linearity=non_linearity
            )
        # Convert lifting to a complex ChannelMLP if self.complex_data==True
        if self.complex_data:
            self.lifting = ComplexValued(self.lifting)
        
        self.projection = ChannelMLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )
        if self.complex_data:
            self.projection = ComplexValued(self.projection)

        self.convgru = CGRU_cell(use_checkpoint=False,
                    input_channels=self.hidden_channels,
                    filter_size=1,
                    num_features=self.hidden_channels)
        
        # 缓存最近一次该样本/序列的隐藏状态（每层一个）
        self._state_cache = {}   # dict: cache_key -> List[Tensor] (len = n_layers)
        self._last_t = {}        # dict: cache_key -> int (上次缓存对应的 t)

        # ==========================================
        # === 新增：针对微调任务的参数冻结逻辑 ===
        # ==========================================
        self.finetune_lifting_only = finetune_lifting_only
        print(f"self.finetune_lifting_only:{self.finetune_lifting_only}")
        if self.finetune_lifting_only:
            # 1. 强制冻结网络中的所有参数 (包含 positional_embedding, fno_blocks, convgru, projection)
            for param in self.parameters():
                param.requires_grad = False
            
            # 2. 仅将 lifting 层的参数解冻
            for param in self.lifting.parameters():
                param.requires_grad = True
            
            print("🚀 [Transfer Learning Mode] 仅解冻 'lifting' 层参数，其它所有核心算子 (FNO, ConvGRU) 已冻结。")
            
    def forward(self, ind, x, device, output_shape=None, init_shape=None, cache_key=None, use_cached_state=True, eval=False):
        (B,H,W,K) = init_shape
        C = self.hidden_channels
    
        if output_shape is None:
            output_shape = [None]*self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None]*(self.n_layers - 1) + [output_shape]

        # 1) 初始化隐状态
        shape = (B,C,H,W)
        prev_hid_state_list = initialize_states(self.n_layers, shape, device)

        # print(f"ind:{ind}")

        # 2) 优先：若有缓存且正好对齐 (last_t == ind-1)，直接复用
        used_cache = False
        if use_cached_state and cache_key is not None and ind > 0:
            last_t = self._last_t.get(cache_key, None)
            if last_t is not None and last_t == ind - 1 and cache_key in self._state_cache:
                # 深拷 + detach，避免跨窗口的计算图牵连
                prev_hid_state_list = [s.detach().clone() for s in self._state_cache[cache_key]]
                used_cache = True
                # print("use cache")

        # 3) 若没有可用缓存，退回原来的预热流程
        if (not used_cache) and ind > 0:
            print(f"prewarming")
            prev_hid_state_list = self.prewarming(ind, x, prev_hid_state_list, output_shape, device)

        # 4) 正常滚动 T 步
        outputs = []
        for t in range(ind, ind + K):
            st = time.time()
            input_t = preprocess_inputs(t, x, device)
        
            out_t, prev_hid_state_list = self.forward_step(input_t[:, :, :, :, 0], prev_hid_state_list, output_shape)
            if eval:
                print(f"eval t:{t}, step time: {time.time()-st}")

            outputs.append(out_t)

        outputs = torch.cat(outputs, dim=-1)  # (B, Cout, H, W, T)


        # 5) 把“本窗口最后时刻”的每层隐状态缓存起来，供下个窗口 (t+1) 直接用
        if cache_key is not None:
            self._state_cache[cache_key] = [s.detach() for s in prev_hid_state_list]
            self._last_t[cache_key] = ind + K - 1

        return outputs

    def forward_step(self, x, prev_hid_state_list, output_shape,**kwargs):
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for layer_idx in range(self.n_layers):
            
            x = self.convgru(x, prev_hid_state_list[layer_idx])

            # fno
            x = self.fno_blocks(
                x, layer_idx, output_shape=output_shape[layer_idx])
            
            prev_hid_state_list[layer_idx] = x

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        x = self.projection(x)
        
        x = x.unsqueeze(-1)  # -> (B,C,H,W,1)

        return x, prev_hid_state_list
    
    def prewarming(self,ind, x, prev_hid_state_list, output_shape,device):

        with torch.no_grad():
            for t in range(0, ind):
                input_t = preprocess_inputs(t, x, device)
                _, prev_hid_state_list = self.forward_step(input_t[:, :, :, :, 0], prev_hid_state_list, output_shape)

        return prev_hid_state_list
    
    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.fno_blocks.n_modes = n_modes
        self._n_modes = n_modes


class FNO1d(FNO):
    """1D Fourier Neural Operator

    For the full list of parameters, see :class:`neuralop.models.FNO`.

    Parameters
    ----------
    modes_height : int
        number of Fourier modes to keep along the height
    """

    def __init__(
        self,
        n_modes_height,
        hidden_channels,
        in_channels=3,
        out_channels=1,
        lifting_channels=256,
        projection_channels=256,
        max_n_modes=None,
        n_layers=4,
        resolution_scaling_factor=None,
        non_linearity=F.gelu,
        stabilizer=None,
        complex_data=False,
        fno_block_precision="full",
        channel_mlp_dropout=0,
        channel_mlp_expansion=0.5,
        norm=None,
        skip="soft-gating",
        separable=False,
        preactivation=False,
        factorization=None,
        rank=1.0,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        domain_padding=None,
        domain_padding_mode="one-sided",
        **kwargs
    ):
        super().__init__(
            n_modes=(n_modes_height,),
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            resolution_scaling_factor=resolution_scaling_factor,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            complex_data=complex_data,
            fno_block_precision=fno_block_precision,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            max_n_modes=max_n_modes,
            norm=norm,
            skip=skip,
            separable=separable,
            preactivation=preactivation,
            factorization=factorization,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            domain_padding=domain_padding,
            domain_padding_mode=domain_padding_mode,
        )
        self.n_modes_height = n_modes_height


class FNO2d(FNO):
    """2D Fourier Neural Operator

    For the full list of parameters, see :class:`neuralop.models.FNO`.

    Parameters
    ----------
    n_modes_width : int
        number of modes to keep in Fourier Layer, along the width
    n_modes_height : int
        number of Fourier modes to keep along the height
    """

    def __init__(
        self,
        n_modes_height,
        n_modes_width,
        hidden_channels,
        in_channels=3,
        out_channels=1,
        lifting_channels=256,
        projection_channels=256,
        n_layers=4,
        resolution_scaling_factor=None,
        max_n_modes=None,
        non_linearity=F.gelu,
        stabilizer=None,
        complex_data=False,
        fno_block_precision="full",
        channel_mlp_dropout=0,
        channel_mlp_expansion=0.5,
        norm=None,
        skip="soft-gating",
        separable=False,
        preactivation=False,
        factorization=None,
        rank=1.0,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        domain_padding=None,
        domain_padding_mode="one-sided",
        **kwargs
    ):
        super().__init__(
            n_modes=(n_modes_height, n_modes_width),
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            resolution_scaling_factor=resolution_scaling_factor,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            complex_data=complex_data,
            fno_block_precision=fno_block_precision,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            max_n_modes=max_n_modes,
            norm=norm,
            skip=skip,
            separable=separable,
            preactivation=preactivation,
            factorization=factorization,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            domain_padding=domain_padding,
            domain_padding_mode=domain_padding_mode,
        )
        self.n_modes_height = n_modes_height
        self.n_modes_width = n_modes_width


class FNO3d(FNO):
    """3D Fourier Neural Operator

    For the full list of parameters, see :class:`neuralop.models.FNO`.

    Parameters
    ----------
    modes_width : int
        number of modes to keep in Fourier Layer, along the width
    modes_height : int
        number of Fourier modes to keep along the height
    modes_depth : int
        number of Fourier modes to keep along the depth
    """

    def __init__(
        self,
        n_modes_height,
        n_modes_width,
        n_modes_depth,
        hidden_channels,
        in_channels=3,
        out_channels=1,
        lifting_channels=256,
        projection_channels=256,
        n_layers=4,
        resolution_scaling_factor=None,
        max_n_modes=None,
        non_linearity=F.gelu,
        stabilizer=None,
        complex_data=False,
        fno_block_precision="full",
        channel_mlp_dropout=0,
        channel_mlp_expansion=0.5,
        norm=None,
        skip="soft-gating",
        separable=False,
        preactivation=False,
        factorization=None,
        rank=1.0,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        domain_padding=None,
        domain_padding_mode="one-sided",
        **kwargs
    ):
        super().__init__(
            n_modes=(n_modes_height, n_modes_width, n_modes_depth),
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            resolution_scaling_factor=resolution_scaling_factor,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            complex_data=complex_data,
            fno_block_precision=fno_block_precision,
            max_n_modes=max_n_modes,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            norm=norm,
            skip=skip,
            separable=separable,
            # Stokes equation for a le,
            preactivation=preactivation,
            factorization=factorization,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            domain_padding=domain_padding,
            domain_padding_mode=domain_padding_mode,
        )
        self.n_modes_height = n_modes_height
        self.n_modes_width = n_modes_width
        self.n_modes_depth = n_modes_depth


def partialclass(new_name, cls, *args, **kwargs):
    """Create a new class with different default values

    Notes
    -----
    An obvious alternative would be to use functools.partial
    >>> new_class = partial(cls, **kwargs)

    The issue is twofold:
    1. the class doesn't have a name, so one would have to set it explicitly:
    >>> new_class.__name__ = new_name

    2. the new class will be a functools object and one cannot inherit from it.

    Instead, here, we define dynamically a new class, inheriting from the existing one.
    """
    __init__ = partialmethod(cls.__init__, *args, **kwargs)
    new_class = type(
        new_name,
        (cls,),
        {
            "__init__": __init__,
            "__doc__": cls.__doc__,
            "forward": cls.forward,
        },
    )
    return new_class


TFNO = partialclass("TFNO", FNO, factorization="Tucker")
TFNO1d = partialclass("TFNO1d", FNO1d, factorization="Tucker")
TFNO2d = partialclass("TFNO2d", FNO2d, factorization="Tucker")
TFNO3d = partialclass("TFNO3d", FNO3d, factorization="Tucker")


def log_mem(tag):
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated()/1024**3
    reserv = torch.cuda.memory_reserved()/1024**3
    print(f"[{tag}] alloc={alloc:.2f} GB, reserved={reserv:.2f} GB")



def check_nan(name, tensor):
    if torch.isnan(tensor).any():
        print(f"⚠️ NaN detected in {name} at positions:", torch.isnan(tensor).nonzero(as_tuple=True))
        print(f"{name} values:", tensor[torch.isnan(tensor)])
    else:
        print(f"{name} OK (no NaN)")



def get_past_rainfall(rainfall, t, nums, H, W):
    """
    Extracts a slice of past rainfall data from BC11S format.

    Parameters:
    - rainfall: Tensor with shape (B, C, 1, 1, S)
    - t: Current time index.
    - nums: Number of timesteps to retrieve.
    - H: Height of the data (expand to this size).
    - W: Width of the data (expand to this size).

    Returns:
    - Tensor with shape (B, C, H, W, nums)
    """
    B, C, _, _, S = rainfall.shape
    start_idx = max(0, t - nums + 1)
    end_idx = min(t + 1, S)

    # 初始化结果 (B, C, H, W, nums)
    extracted_rainfall = torch.zeros(
        (B, nums, H, W, 1), device=rainfall.device
    )

    actual_num_steps = end_idx - start_idx

    # 取出 BC11actual_num_steps
    extracted_data = rainfall[:, 0,0,0, start_idx:end_idx]  # (B, C)
    # 扩展到 BCHWT
    extracted_data = extracted_data.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, H, W, -1)

    # BCHWT
    extracted_rainfall[:,nums - actual_num_steps:,...] = extracted_data

    return extracted_rainfall



def add_border_walls_gpu(dem, wall_height=50):
    """
    Add border walls to a 2D matrix (DEM).

    Parameters:
    - dem: 2D NumPy array, the input matrix.
    - wall_height: Height of the border walls.

    Returns:
    - Updated 2D matrix with border walls.
    """
    wall = wall_height * torch.ones_like(dem)
    wall[:, :, 1:-1, 1:-1] = dem[:, :, 1:-1, 1:-1]

    wall[wall > wall_height] = wall_height

    # import matplotlib.pyplot as plt
    # plt.imshow(wall[0, 0], cmap="jet")
    # plt.savefig("wall.png", dpi=150)

    return wall


def check_tensor(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN found in {name}")
    else:
        print(f"NaN not found in {name}")
    if torch.isinf(tensor).any():
        print(f"Inf found in {name}")



def preprocess_inputs(t, inputs, device, nums=6):
    # BCHWT
    absolute_DEM = MinMaxScaler(
        inputs["absolute_DEM"], inputs["max_DEM"][0], inputs["min_DEM"][0])
    # check_nan("absolute_DEM", absolute_DEM)

    # manhole = MinMaxScaler(inputs["manhole"], 1, 0)
    # check_nan("manhole", manhole) 

    H, W = inputs["absolute_DEM"].shape[2:4]

    # BCHWT
    # rainfall = get_past_rainfall(inputs["rainfall"], t, nums, H, W)
    rainfall = get_past_rainfall_hw(inputs["rainfall"], t, nums)
    # plot_rainfall_hw_at_t0(rainfall, b=0, c=0, title="t=0 rainfall", save_path="rain_t0.png")
    # check_nan("rainfall", rainfall)
    # print(f"rainfall.max():{rainfall.max()}")

    # cumsum_rainfall = get_past_rainfall(inputs["cumsum_rainfall"], t, nums, H, W)
    cumsum_rainfall = get_past_rainfall_hw(inputs["cumsum_rainfall"], t, nums)
    # plot_rainfall_hw_at_t0(rainfall, b=0, c=0, title="t=0 rainfall", save_path="cumrain_t0.png")
    # import sys
    # sys.exit(0)
    # check_nan("cumsum_rainfall", cumsum_rainfall)

    norm_rainfall = MinMaxScaler(rainfall, 6*1.5, 0)
    # check_nan("norm_rainfall", norm_rainfall)

    norm_cumsum_rainfall = MinMaxScaler(cumsum_rainfall, 250*1.5, 0)
    # check_nan("norm_cumsum_rainfall", norm_cumsum_rainfall)
    
    # print(f"rainfall.shape:{rainfall.shape}")
    # print(f"cumsum_rainfall.shape:{cumsum_rainfall.shape}")
    # print(f"absolute_DEM.shape:{absolute_DEM.shape}")
    
    processed_inputs = torch.cat(
        [
            norm_rainfall,
            norm_cumsum_rainfall,
            absolute_DEM,
            # manhole
        ],
        dim=1,
    ).to(dtype=torch.float)

    # check_nan("processed_inputs", processed_inputs)
    # print(f"processed_inputs.shape:{processed_inputs.shape}")

    return processed_inputs




def get_past_rainfall(rainfall, t, nums, H, W):
    """
    Extracts a slice of past rainfall data from BC11S format.

    Parameters:
    - rainfall: Tensor with shape (B, C, 1, 1, S)
    - t: Current time index.
    - nums: Number of timesteps to retrieve.
    - H: Height of the data (expand to this size).
    - W: Width of the data (expand to this size).

    Returns:
    - Tensor with shape (B, C, H, W, nums)
    """
    B, C, _, _, S = rainfall.shape
    start_idx = max(0, t - nums + 1)
    end_idx = min(t + 1, S)

    # 初始化结果 (B, C, H, W, nums)
    extracted_rainfall = torch.zeros(
        (B, nums, H, W, 1), device=rainfall.device
    )

    actual_num_steps = end_idx - start_idx

    # 取出 BC11actual_num_steps
    extracted_data = rainfall[:, 0,0,0, start_idx:end_idx]  # (B, C)
    # 扩展到 BCHWT
    extracted_data = extracted_data.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, H, W, -1)

    # BCHWT
    extracted_rainfall[:,nums - actual_num_steps:,...] = extracted_data

    return extracted_rainfall



def MinMaxScaler(data, max, min):
    """
    Normalizes data using the Min-Max scaling technique.

    Parameters:
    - data: The data tensor to normalize.
    - max: The maximum value for scaling.
    - min: The minimum value for scaling.

    Returns:
    - Normalized data tensor.
    """
    return (data - min) / (max - min)


def r_MinMaxScaler(data, max, min):
    """
    Reverses Min-Max scaling to original values based on the provided maximum and minimum values used in scaling.

    Parameters:
    - data: Normalized data tensor.
    - max: The maximum value used in the original normalization.
    - min: The minimum value used in the original normalization.

    Returns:
    - Original data tensor.
    """
    return data * (max - min) + min


def get_past_rainfall_hw(rainfall, t, nums):
    """
    提取过去 nums 个时刻的降雨数据 (HW 上有空间信息，不需填充)。

    参数:
    - rainfall: Tensor with shape (B, C, H, W, S)
    - t: 当前时刻索引
    - nums: 要提取的时间步数量

    返回:
    - Tensor with shape (B, C, H, W, nums)
    """
    B, C, H, W, S = rainfall.shape
    start_idx = max(0, t - nums + 1)
    end_idx = min(t + 1, S)

    actual_num_steps = end_idx - start_idx

    # 初始化 (B, nums, H, W, C)
    extracted_rainfall = torch.zeros((B, nums, H, W, 1), device=rainfall.device)

    # 提取真实存在的时间片段 (B, C, H, W, actual_num_steps)
    extracted_data = rainfall[:, :, :, :, start_idx:end_idx]
    # extracted_data -> BTHWC
    extracted_data = extracted_data.permute(0, 4, 2, 3, 1)

    # 填充到输出的最后 actual_num_steps 个时间位置
    extracted_rainfall[:, nums-actual_num_steps:,... ] = extracted_data
    

    return extracted_rainfall


def plot_rainfall_hw_at_t0(rainfall, b=0, c=0, title=None, cmap="viridis",
                           save_path=None, show=True):
    """
    绘制 BCHWT 形状的 rainfall 在 t=0 时刻的 H-W 热力图。

    Parameters
    ----------
    rainfall : torch.Tensor or np.ndarray
        形状 (B, C, H, W, T) 的降雨张量/数组。
    b, c : int
        选择的 batch 与 channel 索引。
    title : str or None
        图标题；None 则自动生成。
    cmap : str
        matplotlib colormap 名称。
    save_path : str or None
        若给出，将图保存到该路径。
    show : bool
        是否 plt.show()；在无图形界面环境可设为 False。

    Returns
    -------
    hw : np.ndarray
        取出的 (H, W) 数据（已转成 numpy）。
    """
    import numpy as np
    import matplotlib.pyplot as plt

    try:
        import torch
        is_torch = isinstance(rainfall, torch.Tensor)
    except ImportError:
        is_torch = False

    if is_torch:
        hw = rainfall[b, c, :, :, 0].detach().cpu().numpy()
    else:
        hw = np.asarray(rainfall)[b, c, :, :, 0]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(hw, origin="lower", cmap=cmap)
    ax.set_title(title or f"Rainfall heatmap (b={b}, c={c}, t=0)")
    ax.set_xlabel("W")
    ax.set_ylabel("H")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Rainfall (units)")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return hw
