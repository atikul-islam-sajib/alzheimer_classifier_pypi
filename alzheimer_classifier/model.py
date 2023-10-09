import torch
import torch.nn as nn

class classifier(nn.Module):
  """
    A neural network classifier composed of convolutional and fully connected layers.

    Args:
        input_features (int): The number of input features, default is None.
        output_features (int): The number of output features, default is None.

    Attributes:
        LEFT_CONV (nn.Sequential): Left convolutional layers.
        MIDDLE_CONV (nn.Sequential): Middle convolutional layers.
        RIGHT_CONV (nn.Sequential): Right convolutional layers.
        FC_LAYER (nn.Sequential): Fully connected layers before branching into left, middle, and right.
        LEFT_FC (nn.Sequential): Left fully connected layers.
        MIDD_FC (nn.Sequential): Middle fully connected layers.
        RIGHT_FC (nn.Sequential): Right fully connected layers.

  """
  def __init__(self, input_features = None, output_features = None):
    super().__init__()

    self.LEFT_CONV   = self.left_conv_layer()
    self.MIDDLE_CONV = self.middle_conv_layer()
    self.RIGHT_CONV  = self.right_conv_layer()

    self.FC_LAYER = self.fully_connected_layer()
    self.LEFT_FC  = self.left_fc_layer()
    self.MIDD_FC  = self.middle_fc_layer()
    self.RIGHT_FC = self.right_fc_layer()

  def left_conv_layer(self):
    """
    Defines a sequence of convolutional and pooling layers in PyTorch.

    This sequence consists of three convolutional layers, each followed by a ReLU activation function
    and a max-pooling layer with 2x2 kernel size and stride 2, effectively reducing spatial dimensions.
    The convolutional layers have varying numbers of input and output channels, kernel sizes, and padding.

    - 1st Convolutional Layer:
        - Input channels: 3 (assumes RGB image)
        - Output channels: 32
        - Kernel size: 3x3
        - Padding: 1
    - 2nd Convolutional Layer:
        - Input channels: 32
        - Output channels: 16
        - Kernel size: 3x3
        - Padding: 1
    - 3rd Convolutional Layer:
        - Input channels: 16
        - Output channels: 8
        - Kernel size: 3x3
        - Padding: 1
    """
    return nn.Sequential(
        nn.Conv2d(in_channels  = 3,
                  out_channels = 32,
                  kernel_size  = (3, 3),
                  stride  = 1,
                  padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = (2, 2),
                     stride = (2, 2)),

        nn.Conv2d(in_channels  = 32,
                  out_channels = 16,
                  kernel_size  = (3, 3),
                  stride  = 1,
                  padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = (2, 2),
                     stride = (2, 2)),

        nn.Conv2d(in_channels  = 16,
                  out_channels = 8,
                  kernel_size  = (3, 3),
                  stride  = 1,
                  padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = (2, 2),
                     stride = (2, 2))
    )

  def middle_conv_layer(self):
    """
    Defines a sequence of convolutional and pooling layers in PyTorch.

    This sequence consists of three convolutional layers, each followed by a ReLU activation function
    and a max-pooling layer with 2x2 kernel size and stride 2, effectively reducing spatial dimensions.
    The convolutional layers have varying numbers of input and output channels, kernel sizes, and padding.

    - 1st Convolutional Layer:
        - Input channels: 3 (assumes RGB image)
        - Output channels: 32
        - Kernel size: 4x4
        - Padding: 1
    - 2nd Convolutional Layer:
        - Input channels: 32
        - Output channels: 16
        - Kernel size: 4x4
        - Padding: 1
    - 3rd Convolutional Layer:
        - Input channels: 16
        - Output channels: 8
        - Kernel size: 4x4
        - Padding: 1
    """
    return nn.Sequential(
        nn.Conv2d(in_channels  = 3,
                  out_channels = 32,
                  kernel_size  = (4, 4),
                  stride  = 1,
                  padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = (2, 2),
                     stride = (2, 2)),

        nn.Conv2d(in_channels  = 32,
                  out_channels = 16,
                  kernel_size  = (4, 4),
                  stride  = 1,
                  padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = (2, 2),
                     stride = (2, 2)),

        nn.Conv2d(in_channels  = 16,
                  out_channels = 8,
                  kernel_size  = (4, 4),
                  stride  = 1,
                  padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = (2, 2),
                     stride = (2, 2))
    )

  def right_conv_layer(self):
    """
    Defines a sequence of convolutional and pooling layers in PyTorch.

    This sequence consists of three convolutional layers, each followed by a ReLU activation function
    and a max-pooling layer with 2x2 kernel size and stride 2, effectively reducing spatial dimensions.
    The convolutional layers have varying numbers of input and output channels, kernel sizes, and padding.

    - 1st Convolutional Layer:
        - Input channels: 3 (assumes RGB image)
        - Output channels: 32
        - Kernel size: 5x5
        - Padding: 1
    - 2nd Convolutional Layer:
        - Input channels: 32
        - Output channels: 16
        - Kernel size: 5x5
        - Padding: 1
    - 3rd Convolutional Layer:
        - Input channels: 16
        - Output channels: 8
        - Kernel size: 5x5
        - Padding: 1
    """
    return nn.Sequential(
        nn.Conv2d(in_channels  = 3,
                  out_channels = 32,
                  kernel_size  = (5, 5),
                  stride  = 1,
                  padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = (2, 2),
                     stride = (2, 2)),

        nn.Conv2d(in_channels  = 32,
                  out_channels = 16,
                  kernel_size  = (5, 5),
                  stride  = 1,
                  padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = (2, 2),
                     stride = (2, 2)),

        nn.Conv2d(in_channels  = 16,
                  out_channels = 8,
                  kernel_size  = (5, 5),
                  stride  = 1,
                  padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = (2, 2),
                     stride = (2, 2))
    )

  def fully_connected_layer(self):
    return nn.Sequential(
        nn.Linear(in_features  = 15 * 15 * 8 + 14 * 14 * 8 + 13 * 13 * 8,
                  out_features = 256),

        nn.LeakyReLU()
    )

  def left_fc_layer(self):
    """
    Defines a sequence of fully connected layers in PyTorch.

    This sequence consists of four fully connected (linear) layers, each followed by a ReLU activation function
    and, in some cases, a dropout layer with a specified dropout probability. The fully connected layers have
    different numbers of input and output features.

    - 1st Fully Connected Layer:
        - Input features: 256
        - Output features: 128
    - ReLU Activation Function
    - 2nd Fully Connected Layer:
        - Input features: 128
        - Output features: 64
    - ReLU Activation Function
    - Dropout Layer (p=0.5): Applies dropout with a probability of 0.5
    - 3rd Fully Connected Layer:
        - Input features: 64
        - Output features: 16
    - ReLU Activation Function
    - Dropout Layer (p=0.5): Applies dropout with a probability of 0.5
    - 4th Fully Connected Layer:
        - Input features: 16
        - Output features: 4
    - Softmax Activation Function: Used for multi-class classification

    Note: The final softmax layer is typically used for classification tasks to obtain class probabilities.
    """
    return nn.Sequential(
        nn.Linear(in_features  = 256,
                  out_features = 128),
        nn.ReLU(),

        nn.Linear(in_features  = 128,
                  out_features = 64),
        nn.ReLU(),
        nn.Dropout(p = 0.5),

        nn.Linear(in_features  = 64,
                  out_features = 16),
        nn.ReLU(),
        nn.Dropout(p = 0.5),

        nn.Linear(in_features  = 16,
                  out_features = 4),
        nn.Softmax()
    )

  def middle_fc_layer(self):
    """
    Defines a sequence of fully connected layers in PyTorch.

    This sequence consists of four fully connected (linear) layers, each followed by a ReLU activation function
    and, in some cases, a dropout layer with a specified dropout probability. The fully connected layers have
    different numbers of input and output features.

    - 1st Fully Connected Layer:
        - Input features: 256
        - Output features: 64
    - ReLU Activation Function
    - 2nd Fully Connected Layer:
        - Input features: 64
        - Output features: 32
    - ReLU Activation Function
    - Dropout Layer (p=0.5): Applies dropout with a probability of 0.5
    - 3rd Fully Connected Layer:
        - Input features: 32
        - Output features: 4
    - Softmax Activation Function: Used for multi-class classification

    Note: The final softmax layer is typically used for classification tasks to obtain class probabilities.
    """
    return nn.Sequential(
        nn.Linear(in_features  = 256,
                  out_features = 64),
        nn.ReLU(),
        nn.Dropout(p = 0.5),

        nn.Linear(in_features  = 64,
                  out_features = 32),
        nn.ReLU(),
        nn.Dropout(p = 0.5),

        nn.Linear(in_features  = 32,
                  out_features = 4),
        nn.Softmax()
    )

  def right_fc_layer(self):
    """
    Defines a sequence of fully connected layers in PyTorch.

    This sequence consists of four fully connected (linear) layers, each followed by a ReLU activation function
    and, in some cases, a dropout layer with a specified dropout probability. The fully connected layers have
    different numbers of input and output features.

    - 1st Fully Connected Layer:
        - Input features: 256
        - Output features: 32
    - ReLU Activation Function
    - 2nd Fully Connected Layer:
        - Input features: 64
        - Output features: 16
    - ReLU Activation Function
    - Dropout Layer (p=0.5): Applies dropout with a probability of 0.5
    - 3rd Fully Connected Layer:
        - Input features: 16
        - Output features: 4
    - Softmax Activation Function: Used for multi-class classification

    Note: The final softmax layer is typically used for classification tasks to obtain class probabilities.
    """
    return nn.Sequential(
        nn.Linear(in_features  = 256,
                  out_features = 32),
        nn.ReLU(),
        nn.Dropout(p = 0.3),

        nn.Linear(in_features  = 32,
                  out_features = 16),
        nn.ReLU(),
        nn.Dropout(p = 0.3),

        nn.Linear(in_features  = 16,
                  out_features = 4),
        nn.Softmax()
    )
  def forward(self, x):
    """
    Forward pass of the classifier neural network.

    This method defines the forward pass of the classifier neural network, which consists of the following steps:

    1. Applying the LEFT_CONV, MIDDLE_CONV, and RIGHT_CONV convolutional layers to the input tensor x to extract features.
    2. Reshaping the output of each convolutional branch to have a 1D representation.
    3. Concatenating the flattened outputs of the three branches along dimension 1 to combine features.
    4. Applying the FC_LAYER, a fully connected layer, to the concatenated feature representation.
    5. Branching into LEFT_FC, MIDDLE_FC, and RIGHT_FC fully connected layers for different tasks.
    6. Returning the output of each branch.

    Args:
        x (torch.Tensor): The input tensor to the network.

    Returns:
        tuple: A tuple containing the outputs of the LEFT_FC, MIDDLE_FC, and RIGHT_FC branches.

    Note:
        - LEFT_FC, MIDDLE_FC, and RIGHT_FC are typically used for different subtasks or classes.
        - The concatenation allows the network to capture joint information from the three convolutional branches.
    """
    LEFT   = self.LEFT_CONV(x)
    MIDDLE = self.MIDDLE_CONV(x)
    RIGHT  = self.RIGHT_CONV(x)

    LEFT   = LEFT.reshape(LEFT.shape[0], -1)
    MIDDLE = MIDDLE.reshape(MIDDLE.shape[0], -1)
    RIGHT  = RIGHT.reshape(LEFT.shape[0], -1)

    CONCAT = torch.cat((LEFT, MIDDLE, RIGHT), dim = 1)

    FC_LAYER = self.FC_LAYER(CONCAT)

    LEFT_FC   = self.LEFT_FC(FC_LAYER)
    MIDDLE_FC = self.MIDD_FC(FC_LAYER)
    RIGHT_FC  = self.RIGHT_FC(FC_LAYER)

    return LEFT_FC, MIDDLE_FC, RIGHT_FC

  def model_details(self):
      # Create a classifier model with specified input and output features
      model = classifier(input_features = 3, output_features = 4)
      
      # Print model details header
      print("\t" * 5, " Model Details ","\n")
      print("\t" * 2,"_" * 80, '\n')
      
      # Print model parameters
      print(model.parameters)
      print("\t" * 5,"Model Total Trainable Parameters ","\n")
      print("\t" * 2,"_" * 80, '\n')
      
      # Call the _trainable_parameters method to count and print trainable parameters
      self._trainable_parameters(model = model)
      
  def _trainable_parameters(self, model = None):
      """
        Count and print the trainable parameters in a PyTorch model.

        Args:
            model (nn.Module): The PyTorch model to count parameters for.

        Returns:
            int: The total number of trainable parameters.
      """
      TOTAL_PARAMS = 0
      for layer_name, params in model.named_parameters():
        if params.requires_grad:
            print("Layer # {} & Trainable Parameters # {} ".format(layer_name, params.numel()),'')
            TOTAL_PARAMS+=params.numel()
      
      print("\t" * 2, "_"*80, "\n")
      print("\t" * 5, "Total Trainable Parameters # {} ".format(TOTAL_PARAMS).upper())
      print("\t" * 2, "_"*80, "\n")
      


if __name__ == "__main__":
    model = classifier(input_features = 3, output_features = 4)
    model.model_details()
    
    
    