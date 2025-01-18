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

def pooling_layer(kernel_size, output_height, output_width, channels):
    """
    Calculate FLOPs for a pooling layer.
    Args:
    - kernel_size (int): Size of the pooling window
    - output_height (int): Height of the output feature map
    - output_width (int): Width of the output feature map
    - channels (int): Number of channels
    
    Returns:
    - flops (int): Number of FLOPs
    """
    flops = kernel_size ** 2 * output_height * output_width * channels
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


