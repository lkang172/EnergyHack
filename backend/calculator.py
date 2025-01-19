
class Calculator: 
    def __init__(self, intToParams, batch_size):     
        self.intToParams = intToParams
        self.input_length = 10000
        self.output_size = 128
        self.batch_size = batch_size
        self.layerToInt = {"Linear": 0, "Conv1d": 1, "Conv2d": 2, "Conv3d": 3, "MaxPool1d": 4, "MaxPool2d": 5, "MaxPool3d": 6, "AvgPool1d": 7, "AvgPool2d": 8, "AvgPool3d": 9, 
            "RNN": 10, "LSTM" : 11, "GRU": 12, "ReLU": 13, "Sigmoid" : 14, "Tanh" : 15, "BatchNorm1d": 16, "BatchNorm2d": 17, "LayerNorm": 18,
            "Dropout": 19, "Dropout2d": 20, "Dropout3d": 21, "flatten": 22, "Embedding": 23, "CrossEntropyLoss": 24, "MSELoss": 25, "SmoothL1Loss": 26, 
            "NLLLoss": 27, "HingeEmbeddingLoss": 28, "KLDivLoss": 29, "BCELoss": 30}
        self.total_flops = 0

    """
        Calculate FLOPs for a dense layer.
        Args:
        - input_size (int): Number of input features
        - output_size (int): Number of output features

        Returns:
        - flops (int): Number of FLOPs
    """
    def dense_layer(self, input_size, output_size=128):
        self.output_size = output_size
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

    def conv_layer_1d(self, input_channels, output_channels, kernel_size, stride=1, padding=0):
        self.output_size = output_channels
        output_length = ((self.input_length - kernel_size + 2 * padding) // stride) + 1
        macs = kernel_size * input_channels * output_length * output_channels
        flops = 2 * macs

        return flops

    def conv_layer_2d(self, input_channels, output_channels, kernel_size, stride=1, padding=0):
        self.output_size = output_channels
        output_length = ((self.input_length - kernel_size + 2 * padding) // stride) + 1
        macs = (kernel_size ** 2) * input_channels * output_length * output_channels
        flops = 2 * macs

        return flops

    def conv_layer_3d(self, input_channels, output_channels, kernel_size, stride=1, padding=0):
        self.output_size = output_channels
        output_length = ((self.input_length - kernel_size + 2 * padding) // stride) + 1
        macs = (kernel_size ** 3) * input_channels * output_length * output_channels
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

    def pooling_layer_1d(self, kernel_size, stride=1, padding=0):
        output_length = ((self.input_length - kernel_size + 2 * padding) // stride) + 1
        flops = kernel_size * output_length * 2

        return flops

    def pooling_layer_2d(self, kernel_size, stride=1, padding=0):
        output_length = ((self.input_length - kernel_size + 2 * padding) // stride) + 1
        flops = kernel_size*2 * output_length * 2

        return flops

    def pooling_layer_3d(self, kernel_size, stride=1, padding=0):
        output_length = ((self.input_length - kernel_size + 2 * padding) // stride) + 1
        flops = kernel_size**3 * output_length * 2

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
    def rnn_layer(self, input_size, hidden_size, seq_length):
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
    def lstm_layer(self, input_size, hidden_size, seq_length):

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
    def gru_layer(self, input_size, hidden_size, seq_length):
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

    def relu_activation(self):
        flops = self.output_size  # 1 FLOP per element
        return flops

    def sigmoid_activation(self):
        flops = 4 * self.output_size  # 4 FLOPs per element
        return flops

    def tanh_activation(self):
        flops = 6 * self.output_size  # 6 FLOPs per element
        return flops

    """
        Calculate the approximate number of FLOPs for a batch normalization layer (forward + backward).

        Args:
        - batch_size (int): Number of samples in the batch.
        - numFeatures (int): Number of features per sample (e.g., channels x spatial dimensions).

        Returns:
        - flops (int): Approximate total FLOPs.
    """

    def batch_norm_1d_flops(self, numFeatures):
        return 2 * self.batch_size * numFeatures * self.input_length

    def batch_norm_2d_flops(self, numFeatures):
        return 2 * self.batch_size * numFeatures **2 * self.input_length

    def batch_norm_3d_flops(self, numFeatures):
        return 2 * self.batch_size * numFeatures ** 3 *self.input_length

    """
        Approximate FLOPs for a Dropout layer (forward + backward).
        
        Args:
        - outputSize (int): Total number of elements in the output.
        
        Returns:
        - flops (int): Number of FLOPs.
    """
    def dropout_layer(self):
        flops = 3 * self.output_size
        return flops


    """
        Flatten typically has no compute cost (it's just a reshape).
        
        Args:
        - inputSize (int): Total number of elements in the input.
        
        Returns:
        - flops (int): Number of FLOPs.
    """
    def flatten_layer(self):
        return 0

    """
        Naive FLOPs for an Embedding layer (forward pass).
        Real embedding ops are often just lookups, but this 
        assumes a naive multiplication-based approach.
        
        Args:
        - batch_size (int): Number of samples in the batch.
        - seqLength (int): Number of tokens or elements in each sample.
        - embedDim (int): Size of each embedding vector.
        
        Returns:
        - flops (int): Number of FLOPs.
    """
    def embedding_layer(self, seqLength, embedDim):
        flops = self.batch_size * seqLength * embedDim
        return flops

    """
        Flatten typically has no compute cost (it's just a reshape).
        
        Args:
        - inputSize (int): Total number of elements in the input.
        
        Returns:
        - flops (int): Number of FLOPs.
    """

    """
        Approximate FLOPs for common loss functions (forward + backward).
        
        Args:
        - lossType (str): 'mse', 'mae', 'crossentropy', 'hinge', or 'kl'
        - batch_size (int): Number of samples in the batch
        - outputSize (int): Number of outputs per sample
        
        Returns:
        - flops (int): Estimated total FLOPs
    """
    def CrossEntropyLoss(self, ):
        pass

    def MSELoss(self, ):
        pass

    def SmoothL1Loss(self, ):
        pass

    def HingeEmbeddingLoss(self, ):
        pass

    def KLDivLoss(self, ):
        pass

    def BCELoss(self, ):
        pass

    def calculate(self):
        for layerValue, layerIndex in self.layerToInt.items():
            print(layerIndex)
            params_list = self.intToParams[layerIndex]
            for params in params_list:
                match layerIndex:
                    case 0:
                        self.total_flops += self.dense_layer(*params)
                    case 1:
                        self.total_flops += self.conv_layer_1d(*params)
                    case 2:
                        self.total_flops += self.conv_layer_2d(*params)
                    case 3:
                        self.total_flops += self.conv_layer_3d(*params)
                    case 4:
                        self.total_flops += self.pooling_layer_1d(*params)
                    case 5:
                        self.total_flops += self.pooling_layer_2d(*params)
                    case 6:
                        self.total_flops += self.pooling_layer_3d(*params)
                    case 7:
                        self.total_flops += self.pooling_layer_1d(*params)
                    case 8:
                        self.total_flops += self.pooling_layer_2d(*params)
                    case 9:
                        self.total_flops += self.pooling_layer_3d(*params)
                    case 10:
                        self.total_flops += self.rnn_layer(*params)
                    case 11:
                        self.total_flops += self.lstm_layer(*params)
                    case 12:
                        self.total_flops += self.gru_layer(*params)
                    case 13:
                        self.total_flops += self.relu_activation(*params)
                    case 14:
                        self.total_flops += self.sigmoid_activation(*params)
                    case 15:
                        self.total_flops += self.tanh_activation(*params)
                    case 16:
                        self.total_flops += self.batch_norm_1d_flops(*params)
                    case 17:
                        self.total_flops += self.batch_norm_2d_flops(*params)
                    case 18:
                        self.total_flops += self.layer_norm_flops(*params)
                    case 19:
                        self.total_flops += self.dropout_layer(*params)
                    case 20:
                       self.total_flops += self.dropout_layer(*params)
                    case 21:
                       self.total_flops += self.dropout_layer(*params)
                    case 22:
                        self.total_flops += self.flatten_layer(*params)
                    case 23:
                        self.total_flops += self.embedding_layer(*params)
                    # case 24:
                    #     self.total_flops += self.CrossEntropyLoss(*params)
                    # case 25:
                    #     self.total_flops += self.MSELoss(*params)
                    # case 26:
                    #     self.total_flops += self.SmoothL1Loss(*params)
                    # case 27:
                    #     self.total_flops += self.NLLLoss(*params)
                    # case 28:
                    #     self.total_flops += self.HingeEmbeddingLoss(*params)
                    # case 29:
                    #     self.total_flops += self.KLDivLoss(*params)

                    # case 30:
                    #     self.total_flops += self.BCELoss(*params)
        return self.total_flops