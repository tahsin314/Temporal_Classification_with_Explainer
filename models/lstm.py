import torch
import torch.nn as nn

class BidirectionalLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, num_channels):
        super(BidirectionalLSTMClassifier, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_channels = num_channels

        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2 * num_channels, num_classes)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Reshape the output for classification
        out = out.contiguous().view(x.size(0), -1)

        # Pass through the fully connected layer
        out = self.fc(out)
        return out

# Example usage:
input_size = 64  # Size of input features
hidden_size = 128  # Number of LSTM units
num_layers = 2  # Number of LSTM layers
num_classes = 10  # Number of output classes
num_channels = 3  # Number of input channels

model = BidirectionalLSTMClassifier(input_size, hidden_size, num_layers, num_classes, num_channels)

# Example input data (batch size = 4, sequence length = 100, input size = 64)
input_data = torch.randn(4, 100, input_size)

# Forward pass
output = model(input_data)

# Display the output size
print("Output Size:", output.size())
