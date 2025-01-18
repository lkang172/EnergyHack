'''

Types of layers:
- linear layers nn.Linear - used for classification and regression
- convolutional layers nn.conv[#]d - used for image recognition
- pooling layers nn.Max/AvgPool - used for reducing input size
- 

'''
def dense_layer(input, output):
    # Calculate the number of MACs for a dense layer: input * output
    macs = input * output
    flops = 2 * macs
    return flops

def convoluted_layer(kernelSize, inputChannels, output_height, output_width, output_channels):
    """
    Calculate the number of MACs and FLOPs for a convolutional layer.
    
    Args:
    - kernelSize (int): Width and height of the kernel (K).
    - inputChannels (int): Number of input channels (C_in).
    - output_height (int): Height of the output feature map (H_out).
    - output_width (int): Width of the output feature map (W_out).
    - output_channels (int): Number of output channels (C_out).
    
    Returns:
    - flops (int): Number of FLOPs.
    """
    # Calculate the number of MACs for the convolutional layer
    macs = (kernelSize ** 2) * inputChannels * output_height * output_width * output_channels
    # Each MAC operation consists of one multiplication and one addition
    flops = 2 * macs
    return flops

# Example usage
kernelSize = 3
inputChannels = 64
output_height = 32
output_width = 32
output_channels = 128

flops = convoluted_layer(kernelSize, inputChannels, output_height, output_width, output_channels)
print(f"FLOPs: {flops}")
