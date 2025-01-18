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

def recurrentLayerFlops(layerType, inputSize, hiddenSize, seqLength):
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
