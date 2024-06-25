import torch
from torch import nn
from torch.nn import functional as F

from collections import OrderedDict


def _make_pair(value):
	if isinstance(value, int):
		value = (value,) * 2
	return value

def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
	"""
	Activation functions for ['relu', 'lrelu', 'prelu'].

	Parameters
	----------
	act_type: str
		one of ['relu', 'lrelu', 'prelu'].
	inplace: bool
		whether to use inplace operator.
	neg_slope: float
		slope of negative region for `lrelu` or `prelu`.
	n_prelu: int
		`num_parameters` for `prelu`.
	----------
	"""
	act_type = act_type.lower()
	if act_type == 'relu':
		layer = nn.ReLU(inplace)
	elif act_type == 'lrelu':
		layer = nn.LeakyReLU(neg_slope, inplace)
	elif act_type == 'prelu':
		layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
	else:
		raise NotImplementedError(
			'activation layer [{:s}] is not found'.format(act_type))
	return layer

def sequential(*args):
	"""
	Modules will be added to the a Sequential Container in the order they
	are passed.
	
	Parameters
	----------
	args: Definition of Modules in order.
	-------

	"""
	if len(args) == 1:
		if isinstance(args[0], OrderedDict):
			raise NotImplementedError(
				'sequential does not support OrderedDict input.')
		return args[0]
	modules = []
	for module in args:
		if isinstance(module, nn.Sequential):
			for submodule in module.children():
				modules.append(submodule)
		elif isinstance(module, nn.Module):
			modules.append(module)
	return nn.Sequential(*modules)

def conv_layer(in_channels,
			   out_channels,
			   kernel_size,
			   bias=True):
	"""
	Re-write convolution layer for adaptive `padding`.
	"""
	kernel_size = _make_pair(kernel_size)
	padding = (int((kernel_size[0] - 1) / 2), 
			   int((kernel_size[1] - 1) / 2))
	return nn.Conv2d(in_channels,
					 out_channels,
					 kernel_size,
					 padding=padding,
					 bias=bias)

def pixelshuffle_block(in_channels,
					   out_channels,
					   upscale_factor=2,
					   kernel_size=3):
	"""
	Upsample features according to `upscale_factor`.
	"""
	conv = conv_layer(in_channels,
					  out_channels * (upscale_factor ** 2),
					  kernel_size)
	pixel_shuffle = nn.PixelShuffle(upscale_factor)
	return sequential(conv, pixel_shuffle)

class ESA(nn.Module):
	"""
	Modification of Enhanced Spatial Attention (ESA), which is proposed by 
	`Residual Feature Aggregation Network for Image Super-Resolution`
	Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
	are deleted.
	"""

	def __init__(self, esa_channels, n_feats, conv):
		super(ESA, self).__init__()
		f = esa_channels
		self.conv1 = conv(n_feats, f, kernel_size=1)
		self.conv_f = conv(f, f, kernel_size=1)
		self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
		self.conv3 = conv(f, f, kernel_size=3, padding=1)
		self.conv4 = conv(f, n_feats, kernel_size=1)
		self.sigmoid = nn.Sigmoid()
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		c1_ = (self.conv1(x))
		c1 = self.conv2(c1_)
		v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
		c3 = self.conv3(v_max)
		c3 = F.interpolate(c3, (x.size(2), x.size(3)),
						   mode='bilinear', align_corners=False)
		cf = self.conv_f(c1_)
		c4 = self.conv4(c3 + cf)
		m = self.sigmoid(c4)
		return x * m

class SELayer(nn.Module):
	def __init__(self, in_channel, out_channel, reduction=16):
		"""
		Initialize the Squeeze-and-Excitation layer.
		
		:param in_channel: Number of input channels.
		:param out_channel: Number of output channels.
		:param reduction: Reduction ratio for controlling the bottleneck layer's channel number.
		"""
		super(SELayer, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Sequential(
			nn.Linear(in_channel, in_channel // reduction, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(in_channel // reduction, out_channel, bias=False),
			nn.Sigmoid()
		)
		
	def forward(self, x):
		"""
		Forward pass of the Squeeze-and-Excitation layer.
		
		:param x: Input tensor of shape (batch_size, in_channel, H, W).
		:return: Output tensor after squeeze and excitation operations.
		"""
		b, c, _, _ = x.size()
		# Squeeze operation
		y = self.avg_pool(x).view(b, c)
		# Excitation operation
		y = self.fc(y).view(b, c, 1, 1)
		# Scale the input feature map
		return x * y.expand_as(x)

class ERLFB(nn.Module):
	"""
	Efficient Residual Local Feature Block(E)

	"""

	def __init__(self, 
				 in_channels, 
				 out_channels, 
				 esa_channels = 16):
		super(ERLFB, self).__init__()
		self.c1_r = nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups= 2)
		self.c2_r = nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups = 4)
		self.c3_r = nn.Conv2d(in_channels,out_channels, 3, 1, 1, groups = 8)
		self.c4_r = nn.Conv2d(in_channels, out_channels, 1, 1)
		self.c5 = conv_layer(in_channels, out_channels, 1)
		self.esa = ESA(esa_channels, in_channels, nn.Conv2d)
		self.se = SELayer(in_channels, out_channels)
		self.act = activation("lrelu", neg_slope=0.05)

	def forward(self, x):
		out = self.c1_r(x)
		out = self.act(out)
		out = self.c2_r(out)
		out = self.act(out)
		out = self.c3_r(out)
		out = self.act(out)
		out = self.c4_r(out)
		out = self.act(out)
		out1 = self.se(out)
		out2 = self.esa(out)
		out = (x + out1 + out2)/3.0
		out = self.c5(out)

		return out 

class EResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, esa_channels = 16, groups=4):
		super(EResidualBlock, self).__init__()

		self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=groups)
		self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=groups)
		self.pw = nn.Conv2d(out_channels, out_channels, 1, 1, 0)
		self.esa = ESA(esa_channels, out_channels, nn.Conv2d)
		self.act = activation("lrelu", neg_slope= 0.05)

	def forward(self, x):
		out = self.act(self.conv1(x))
		out = self.act(self.conv2(out))
		out = self.act(self.pw(out)+x)
		out = self.esa(out)
		return out

class MultiHead(nn.Module):
	"""
	Efficient multihead Spatial Attention
	
	"""
	def __init__(self, 
			  in_channels, 
			  out_channels,
			  num_heads = 4):
		super(MultiHead, self).__init__()
		self.ln = nn.LayerNorm(normalized_shape=[out_channels], elementwise_affine= True)
		self.num_heads = num_heads 
		self.esr_list = nn.ModuleList()
		for i in range(1, self.num_heads + 1):
			self.esr_list.append(EResidualBlock(in_channels= in_channels, out_channels= out_channels, groups= 2**i))
		self.head_mix = conv_layer(out_channels*(self.num_heads+1), out_channels, 1)

	def forward(self, x):
		out = x
		for esr_layer in self.esr_list:
			out = torch.cat((out, esr_layer(x)), dim = 1)
		out = self.head_mix(out) + x
		# Expects the channel to be last layer for normalization.[N, H, W, C]
		out = out.permute(0, 2, 3, 1)
		out = self.ln(out)
		#bring back the original form[N, C, H, W]
		out = out.permute(0, 3, 1, 2)
		return out 

class EMaxGMan(nn.Module):                                                       
	"""                                                                         
	Efficient Multi Head Attention with Multi Group Activation Mixture[EMHAMGA]                                  
	"""                                                                         
																				
	def __init__(self, in_channels = 3, out_channels = 3, feature_channels = 64, upscale = 4):
		super(EMaxGMan, self).__init__()                                                                              
		self.conv_in = conv_layer(in_channels, feature_channels, kernel_size = 3)
		self.multi_head = MultiHead(feature_channels, feature_channels)         
		self.block_1 = ERLFB(feature_channels,feature_channels)
		self.block_2 = ERLFB(feature_channels,feature_channels)
		self.block_3 = ERLFB(feature_channels,feature_channels)

		self.conv_out = conv_layer(feature_channels,
									   feature_channels,
									   kernel_size=3)

		self.upsampler = pixelshuffle_block(feature_channels,
												  out_channels,
												  upscale_factor=upscale)														
																				
	def forward(self, x):                                                       
		expanded_feature = self.conv_in(x)                                      
		out_head = self.multi_head(expanded_feature)
		out_b1 = self.block_1(out_head)
		out_b2 = self.block_2(out_b1)
		out_b3 = self.block_3(out_b2)
		out_low_resolution = self.conv_out(out_b3) + expanded_feature 
		output = self.upsampler(out_low_resolution)
		return output 