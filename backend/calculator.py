

"""
Calculate FLOPs for a multi-head attention layer.
Args:
- input_size (int)
- num_heads (int)
- batch_size (int)
- input_sequence (int)

"""

def multi_head_attention(input_size, num_heads, batch_size):
    vectors = batch_size * input_size

"""
Calculate FLOPs for a dense layer.
Args:
- input_size (int): Number of input features
- output_size (int): Number of output features

Returns:
- flops (int): Number of FLOPs
"""
def dense_layer(input_size, output_size):
    macs = input_size * output_size
    flops = 2 * macs 
    return flops

"""
Calculate FLOPs for a convolutional layer.
Args:
- kernel_size (int): Width and height of the kernel
- input_channels (int): Number of input channels
- output_height (int): Height of the output feature map
- output_width (int): Width of the output feature map
- output_channels (int): Number of output channels

Returns:
- flops (int): Number of FLOPs
"""

def conv_layer_1d(kernel_size, input_channels, output_length, output_channels):
    macs = kernel_size * input_channels * output_length * output_channels
    flops = 2 * macs
    return flops

def conv_layer_2d(kernel_size, input_channels, output_height, output_width, output_channels):
    macs = (kernel_size ** 2) * input_channels * output_height * output_width * output_channels
    flops = 2 * macs
    return flops

def conv_layer_3d(kernel_size, input_channels, output_depth, output_height, output_width, output_channels):
    macs = (kernel_size ** 3) * input_channels * output_depth * output_height * output_width * output_channels
    flops = 2 * macs
    return flops

"""
    Calculate FLOPs for a pooling layer.
    Args:
    - kernel_size (int): Size of the pooling window
    - output_size (tuple): Output size of the feature map (height, width[, depth])
    - channels (int): Number of channels
    - dimensions (int): Number of dimensions (1, 2, or 3)
    
    Returns:
    - flops (int): Number of FLOPs
"""

def pooling_layer_1d(kernel_size, output_length, channels):
    flops = kernel_size * output_length * channels
    return flops

def pooling_layer_2d(kernel_size, output_height, output_width, channels):
    flops = kernel_size ** 2 * output_height * output_width * channels
    return flops

def pooling_layer_3d(kernel_size, output_depth, output_height, output_width, channels):
    flops = kernel_size ** 3 * output_depth * output_height * output_width * channels
    return flops

"""
    Calculate FLOPs for a basic RNN layer.
    Args:
    - input_size (int): Number of input features
    - hidden_size (int): Number of hidden units
    - seq_length (int): Length of the input sequence
    
    Returns:
    - flops (int): Number of FLOPs
"""
def rnn_layer(input_size, hidden_size, seq_length):
    macs = ((input_size * hidden_size) + (hidden_size * hidden_size) + hidden_size) * seq_length
    flops = 2 * macs
    return flops

"""
    Calculate FLOPs for an LSTM layer.
    Args:
    - input_size (int): Number of input features
    - hidden_size (int): Number of hidden units
    - seq_length (int): Length of the input sequence
    
    Returns:
    - flops (int): Number of FLOPs
"""
def lstm_layer(input_size, hidden_size, seq_length):

    macs = 4 * ((input_size * hidden_size) + (hidden_size * hidden_size) + hidden_size) * seq_length
    flops = 2 * macs
    return flops

"""
    Calculate FLOPs for a GRU layer.
    Args:
    - input_size (int): Number of input features
    - hidden_size (int): Number of hidden units
    - seq_length (int): Length of the input sequence
    
    Returns:
    - flops (int): Number of FLOPs
    """
def gru_layer(input_size, hidden_size, seq_length):
    macs = 3 * ((input_size * hidden_size) + (hidden_size * hidden_size) + hidden_size) * seq_length
    flops = 2 * macs
    return flops


"""
Calculate the number of FLOPs for an activation function.

Args:
- activationType (str): Type of activation function ('relu', 'sigmoid', 'tanh').
- outputSize (int): Total number of elements in the output tensor.

Returns:
- flops (int): Number of FLOPs.
"""

def relu_activation(output_size):
    flops = output_size  # 1 FLOP per element
    return flops

def sigmoid_activation(output_size):
    flops = 4 * output_size  # 4 FLOPs per element
    return flops

def tanh_activation(output_size):
    flops = 6 * output_size  # 6 FLOPs per element
    return flops

"""
    Calculate the approximate number of FLOPs for a batch normalization layer (forward + backward).

    Args:
    - batchSize (int): Number of samples in the batch.
    - numFeatures (int): Number of features per sample (e.g., channels x spatial dimensions).

    Returns:
    - flops (int): Approximate total FLOPs.
"""

def batch_norm_1d_flops(batchSize, numFeatures):
    return 2 * batchSize * numFeatures

def batch_norm_2d_flops(batchSize, channels, height, width):
    return 2 * batchSize * channels * height * width

def layer_norm_flops(batchSize, seqLength, embedDim):
    return 2 * batchSize * seqLength * embedDim

"""
    Approximate FLOPs for a Dropout layer (forward + backward).
    
    Args:
    - outputSize (int): Total number of elements in the output.
    
    Returns:
    - flops (int): Number of FLOPs.
"""
def dropout_layer(outputSize):
    flops = 2 * outputSize
    return flops

"""
    Flatten typically has no compute cost (it's just a reshape).
    
    Args:
    - inputSize (int): Total number of elements in the input.
    
    Returns:
    - flops (int): Number of FLOPs.
"""
def flatten_layer(inputSize):
    return 0

"""
    Naive FLOPs for an Embedding layer (forward pass).
    Real embedding ops are often just lookups, but this 
    assumes a naive multiplication-based approach.
    
    Args:
    - batchSize (int): Number of samples in the batch.
    - seqLength (int): Number of tokens or elements in each sample.
    - embedDim (int): Size of each embedding vector.
    
    Returns:
    - flops (int): Number of FLOPs.
"""
def embedding_layer(batchSize, seqLength, embedDim):
    flops = batchSize * seqLength * embedDim
    return flops

"""
    Flatten typically has no compute cost (it's just a reshape).
    
    Args:
    - inputSize (int): Total number of elements in the input.
    
    Returns:
    - flops (int): Number of FLOPs.
"""
def flatten_layer(inputSize):
    return 0

"""
    Approximate FLOPs for common loss functions (forward + backward).
    
    Args:
    - lossType (str): 'mse', 'mae', 'crossentropy', 'hinge', or 'kl'
    - BATCHSIZE (int): Number of samples in the batch
    - outputSize (int): Number of outputs per sample
    
    Returns:
    - flops (int): Estimated total FLOPs
"""
def loss_function_flops(lossType, BATCHSIZE, outputSize):
    lossType = lossType.lower()
    if lossType == 'mse':
        flops_per_element = 4
    elif lossType == 'mae':
        flops_per_element = 4
    elif lossType == 'crossentropy':
        flops_per_element = 6
    elif lossType == 'hinge':
        flops_per_element = 6
    elif lossType == 'kl':
        flops_per_element = 6
    else:
        raise ValueError("Choose from 'mse', 'mae', 'crossentropy', 'hinge', or 'kl'.")

    return flops_per_element * BATCHSIZE * outputSize

layertoInt = {
    "dense_layer": 0,
    "conv_layer_1d": 1,
    "conv_layer_2d": 2,
    "conv_layer_3d": 3,
    "pooling_layer_1d": 4,
    "pooling_layer_2d": 5,
    "pooling_layer_3d": 6,
    "rnn_layer": 7,
    "lstm_layer": 8,
    "gru_layer": 9,
    "relu_activation": 10,
    "sigmoid_activation": 11,
    "tanh_activation": 12,
    "batch_norm_1d_flops": 13,
    "batch_norm_2d_flops": 14,
    "layer_norm_flops": 15,
    "dropout_layer": 16,
    "flatten_layer": 17,
    "embedding_layer": 18,
    "residual_layer": 19,
    "loss_function_flops": 20
}

intToParam = {
    0: [[12, 34], [34, 22], [23, 22]],
    1: [[3, 64, 128, 256], [5, 32, 64, 128]],
    2: [[3, 64, 32, 32, 128], [5, 32, 16, 16, 64]],
    3: [[3, 64, 16, 16, 16, 128], [5, 32, 8, 8, 8, 64]],
    4: [[2, 128, 64], [3, 256, 128]],
    5: [[2, 32, 32, 64], [3, 16, 16, 32]],
    6: [[2, 16, 16, 16, 32], [3, 8, 8, 8, 16]],
    7: [[128, 64, 32], [256, 128, 64]],
    8: [[128, 64, 32], [256, 128, 64]],
    9: [[128, 64, 32], [256, 128, 64]],
    10: [[1000], [2000]],
    11: [[1000], [2000]],
    12: [[1000], [2000]],
    13: [[32, 64], [64, 128]],
    14: [[32, 64, 32, 32], [64, 128, 16, 16]],
    15: [[32, 64, 128], [64, 128, 256]],
    16: [[1000], [2000]],
    17: [[1000], [2000]],
    18: [[32, 64, 128], [64, 128, 256]],
    19: [[1000], [2000]],
    20: [['mse', 32, 64], ['crossentropy', 64, 128]]
}

total_flops = 0

for layer_name, layerIndex in layertoInt.items():
    params_list = intToParam[layerIndex]
    for params in params_list:
        match layerIndex:
            case 0:
                total_flops += dense_layer(*params)
            case 1:
                total_flops += conv_layer_1d(*params)
            case 2:
                total_flops += conv_layer_2d(*params)
            case 3:
                total_flops += conv_layer_3d(*params)
            case 4:
                total_flops += pooling_layer_1d(*params)
            case 5:
                total_flops += pooling_layer_2d(*params)
            case 6:
                total_flops += pooling_layer_3d(*params)
            case 7:
                total_flops += rnn_layer(*params)
            case 8:
                total_flops += lstm_layer(*params)
            case 9:
                total_flops += gru_layer(*params)
            case 10:
                total_flops += relu_activation(*params)
            case 11:
                total_flops += sigmoid_activation(*params)
            case 12:
                total_flops += tanh_activation(*params)
            case 13:
                total_flops += batch_norm_1d_flops(*params)
            case 14:
                total_flops += batch_norm_2d_flops(*params)
            case 15:
                total_flops += layer_norm_flops(*params)
            case 16:
                total_flops += dropout_layer(*params)
            case 17:
                total_flops += flatten_layer(*params)
            case 18:
                total_flops += embedding_layer(*params)
            case 19:
                total_flops += residual_layer(*params)
            case 20:
                total_flops += loss_function_flops(*params)
