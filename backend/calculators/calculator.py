def dense_layer(input_size, output_size):
    """
    Calculate FLOPs for a dense layer.
    Args:
    - input_size (int): Number of input features
    - output_size (int): Number of output features
    
    Returns:
    - flops (int): Number of FLOPs
    """
    macs = input_size * output_size
    flops = 2 * macs 
    return flops

def conv_layer(kernel_size, input_channels, output_height, output_width, output_channels):
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
    macs = (kernel_size ** 2) * input_channels * output_height * output_width * output_channels
    flops = 2 * macs
    return flops

def pooling_layer(kernel_size, output_size, channels, dimensions=2):
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
    if dimensions == 1:
        output_length = output_size[0]
        flops = kernel_size * output_length * channels
    elif dimensions == 2:
        output_height, output_width = output_size
        flops = kernel_size ** 2 * output_height * output_width * channels
    elif dimensions == 3:
        output_depth, output_height, output_width = output_size
        flops = kernel_size ** 3 * output_depth * output_height * output_width * channels
    else:
        raise ValueError("Unsupported number of dimensions")
    
    return flops


def reccurent_layer(layerType, inputSize, hiddenSize, seqLength):
    """
    Calculate the number of FLOPs for a recurrent layer (LSTM or GRU).
    
    Args:
    - layerType (str): Type of recurrent layer ('lstm' or 'gru').
    - inputSize (int): Number of input features.
    - hiddenSize (int): Number of hidden units.
    - seqLength (int): Length of the input sequence.
    
    Returns:
    - flops (int): Number of FLOPs.
    """
    if layerType.lower() == 'lstm':
        macs = 4 * ((inputSize * hiddenSize) + (hiddenSize * hiddenSize) + hiddenSize) * seqLength
    elif layerType.lower() == 'gru':
        macs = 3 * ((inputSize * hiddenSize) + (hiddenSize * hiddenSize) + hiddenSize) * seqLength
    
    flops = 2 * macs
    return flops

def activation_layer(activationType, outputSize):
    """
    Calculate the number of FLOPs for an activation function.
    
    Args:
    - activationType (str): Type of activation function ('relu', 'sigmoid', 'tanh').
    - outputSize (int): Total number of elements in the output tensor.
    
    Returns:
    - flops (int): Number of FLOPs.
    """
    if activationType.lower() == 'relu':
        flops = outputSize  # 1 FLOP per element
    elif activationType.lower() == 'sigmoid':
        flops = 4 * outputSize  # 4 FLOPs per element
    elif activationType.lower() == 'tanh':
        flops = 6 * outputSize  # 6 FLOPs per element
    else:
        raise ValueError("activationType must be 'relu', 'sigmoid', or 'tanh'")
    
    return flops

def batch_norm_layer(batchSize, numFeatures):
    """
    Calculate the approximate number of FLOPs for a batch normalization layer (forward + backward).

    Args:
    - batchSize (int): Number of samples in the batch.
    - numFeatures (int): Number of features per sample (e.g., channels x spatial dimensions).

    Returns:
    - flops (int): Approximate total FLOPs.
    """
    flops = 2 * batchSize * numFeatures
    return flops

def dropout_layer(outputSize):
    """
    Approximate FLOPs for a Dropout layer (forward + backward).
    
    Args:
    - outputSize (int): Total number of elements in the output.
    
    Returns:
    - flops (int): Number of FLOPs.
    """
    flops = 2 * outputSize
    return flops

def flatten_layer(inputSize):
    """
    Flatten typically has no compute cost (it's just a reshape).
    
    Args:
    - inputSize (int): Total number of elements in the input.
    
    Returns:
    - flops (int): Number of FLOPs.
    """
    return 0

def embedding_layer(batchSize, seqLength, embedDim):
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
    flops = batchSize * seqLength * embedDim
    return flops

def residual_layer(outputSize):
    """
    Approximate FLOPs for a simple residual skip connection
    (adding two tensors of the same shape).
    
    Args:
    - outputSize (int): Total number of elements in the output.
    
    Returns:
    - flops (int): Number of FLOPs.
    """
    flops = 2 * outputSize
    return flops

def flatten_layer(inputSize):
    """
    Flatten typically has no compute cost (it's just a reshape).
    
    Args:
    - inputSize (int): Total number of elements in the input.
    
    Returns:
    - flops (int): Number of FLOPs.
    """
    return 0

def loss_function_flops(lossType, BATCHSIZE, outputSize):
    """
    Approximate FLOPs for common loss functions (forward + backward).
    
    Args:
    - lossType (str): 'mse', 'mae', 'crossentropy', 'hinge', or 'kl'
    - BATCHSIZE (int): Number of samples in the batch
    - outputSize (int): Number of outputs per sample
    
    Returns:
    - flops (int): Estimated total FLOPs
    """
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



layertoInt = {"dense_layer": 0, "conv_layer": 1, "pooling_layer": 2, "reccurent_layer": 3, "activation_layer": 4, "batch_norm_layer": 5, "dropout_layer": 6, "flatten_layer": 7, "embedding_layer": 8, "residual_layer": 9, "loss_function_flops": 10}
intToParam = {0: [[12, 34], [34, 22], [23, 22]], 1: [[1, 2, 3, 4, 5], [45, 23, 12, 23, 45]], 2: [[1, 34, 123, 186], [34, 23, 65, 23]], 3: [[1, 2, 3, 4], [23, 23, 23, 23]], 4: [[1, 2], [23, 23]], 5: [[1, 2], [23, 23]], 6: [[1], [23]], 7: [[1], [23]], 8: [[1, 2, 3], [23, 23, 23]], 9: [[1], [23]], 10: [[1, 2, 3], [23, 23, 23]]}
total_flops = 0

for layer_name, layerIndex in layertoInt.items():
    params_list = intToParam[layerIndex]
    for params in params_list:
        match layerIndex:
            case 0:
                total_flops += dense_layer(*params)
            case 1:
                total_flops += conv_layer(*params)
            case 2:
                total_flops += pooling_layer(*params)
            case 3:
                total_flops += reccurent_layer(*params)
            case 4:
                total_flops += activation_layer(*params)
            case 5:
                total_flops += batch_norm_layer(*params)
            case 6:
                total_flops += dropout_layer(*params)
            case 7:
                total_flops += flatten_layer(*params)
            case 8:
                total_flops += embedding_layer(*params)
            case 9:
                total_flops += residual_layer(*params)
            case 10:
                total_flops += loss_function_flops(*params)