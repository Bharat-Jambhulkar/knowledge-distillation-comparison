import torch.nn as nn

class TeacherCNN(nn.Module):
    def __init__(self, input_features, num_classes=10):
        super().__init__()

        num_conv_layers = 2
        num_filters = 64
        kernel_size = 3
        num_fc_layers = 2
        fc_layer_size = 64
        dropout_rate = 0.407066

        # Convolutional Layers 
        conv_layers = []
        in_channels = input_features

        for _ in range(num_conv_layers):
            conv_layers.append(nn.Conv2d(in_channels, num_filters, kernel_size=kernel_size, padding='same'))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.BatchNorm2d(num_filters))
            conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = num_filters  # next layer input

        # Use adaptive pooling to avoid manual size calculation
        conv_layers.append(nn.AdaptiveAvgPool2d((7, 7)))

        self.features = nn.Sequential(*conv_layers)

        # Fully Connected Layers 
        fc_layers = []
        fc_input_size = num_filters * 7 * 7

        fc_layers.append(nn.Flatten())

        for i in range(num_fc_layers):
            fc_layers.append(nn.Linear(fc_input_size if i == 0 else fc_layer_size, fc_layer_size))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(p=dropout_rate))

        # Output layer
        fc_layers.append(nn.Linear(fc_layer_size, num_classes))

        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x