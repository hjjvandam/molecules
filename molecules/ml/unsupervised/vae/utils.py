def conv_output_dim(input_dim, kernel_size, stride, padding,
                      transpose=False):
    """
    Parameters
    ----------
    input_dim : int
        input size. may include padding
    
    kernel_size : int
        filter size

    stride : int
        stride length

    padding : int
        length of 0 pad

    """

    if transpose:
        return (input_dim - 1) * stride + kernel_size - 2*padding

    return (2*padding + input_dim - kernel_size) // stride + 1

def conv_output_shape(input_dim, kernel_size, stride, padding,
                      num_filters, transpose=False, dim=2):
    """
    Parameters
    ----------
    input_dim : int
        input size
    
    kernel_size : int
        filter size

    stride : int
        stride length

    padding : int
        length of 0 pad

    num_filters : int
        number of filters

    transpose : bool
        signifies whether Conv or ConvTranspose

    dim : int
        1 or 2, signifies Conv1d or Conv2d

    Returns
    -------
    (channels, N, N) tuple

    """
    output_dim = conv_output_dim(input_dim, kernel_size, stride,
                                 padding, transpose)

    if dim == 1:
        return num_filters, output_dim
    if dim == 2:
        return num_filters, output_dim, output_dim

    raise ValueError(f'Invalid dim: {dim}')

def same_padding(input_dim, kernel_size, stride):
    """
    Implements Keras-like same padding.
    If the stride is one then use same_padding.
    Otherwise, select the smallest pad such that the
    kernel_size fits evenly within the input_dim.

    Note: Assumes square matrix with dimension input_dim
    """
    if stride == 1:
        # In this case we want output_dim = input_dim
        # input_dim = output_dim = (2*pad + input_dim - kernel_size) // stride + 1
        return (input_dim * (stride - 1) - stride + kernel_size) // 2

    # Largest i such that: alpha = kernel_size + i*stride <= input_dim
    # Then input_dim - alpha is the pad
    # i <= (input_dim - kernel_size) // stride
    for i in reversed(range((input_dim - kernel_size) // stride + 1)):
        alpha = kernel_size + i*stride
        if alpha <= input_dim:
            return input_dim - alpha

    raise Exception(f'No padding found')


# TODO: These pytorch functions should be moved elsewhere
from torch import nn

# TODO: conider moving this into the HyperParams class
#       could make base classes for ModelArchHyperParams
#       which handles layes and can return pytorch layer
#       types and activations. OptimizerHyperParams can
#       return different optimizers.
def get_activation(activation):
    """
    Parameters
    ----------
    activation : str
        type of activation e.g. 'ReLU', etc

    """
    if activation is 'ReLU':
        return nn.ReLU()
    elif activation is 'Sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError(f'Invalid activation type: {activation}')

# TODO: generalize this more.
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif type(m) in [nn.Conv2d, nn.ConvTranspose2d]:
        nn.init.xavier_uniform_(m.weight)
