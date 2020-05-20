def conv2d_output_dim(input_dim, kernel_size, stride, padding,
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

def conv2d_output_shape(input_dim, kernel_size, stride, padding,
                        num_filters, transpose=False):
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


    Returns
    -------
    (N,N, num_filters) tuple

    """
    output_dim = conv2d_output_dim(input_dim, kernel_size, stride,
                                   padding, transpose)
    return (output_dim, output_dim, num_filters)

def _same_padding(input_dim, kernel_size, stride):
    """
    Parameters
    ----------
    n : int
        input size
    
    kernel_size : int
        filter size

    stride : int
        stride length

    Effects
    -------
    In this case we want output_dim = input_dim

    input_dim = output_dim = (2*pad + input_dim - kernel_size) // stride + 1

    Solve for pad. 
    """
    return (input_dim * (stride - 1) - stride + kernel_size) // 2

def same_padding(input_dim, kernel_size, stride):
    """
    Implements Keras-like same padding.
    If the stride is one then use same_padding.
    Otherwise, select the smallest pad such that the
    kernel_size fits evenly within the input_dim.
    """
    if stride == 1:
        return _same_padding(input_dim, kernel_size, stride)

    # TODO: this could have bugs for stride != 1. Needs testing.
    return _same_padding(input_dim // stride, kernel_size, 1)


# TODO: These pytorch functions should be moved elsewhere
from torch import nn

# TODO: conider moving this into the HyperParams class
#       could make base classes for ModelArchHyperParams
#       which handles layes and can return pytorch layer
#       types and activations. OptimizerHyperParams can
#       return different optimizers.
def select_activation(activation):
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
