import torch
from torch import nn
from torch.autograd import Variable
    
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.5, class_weights=None, use_batchnorm=False, loss_type="classic"):
        super(MLPClassifier, self).__init__()

        # Define layers list that will hold each layer dynamically
        layers = []
        in_features = input_size
        self.class_weights = class_weights
        self.loss_type = loss_type
        self.num_classes = output_size
        
        # Create hidden layers dynamically based on hidden_sizes list
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))
                
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            layers.append(nn.Dropout(dropout_rate))
            
            # Update in_features to current hidden_size for next layer
            in_features = hidden_size

        # Output layer
        layers.append(nn.Linear(in_features, output_size))
        
        # Use nn.Sequential to define the full network as a single sequence
        self.model = nn.Sequential(*layers)
        
        self.initialize_weights()
        
        self.first_layer = None
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                self.first_layer = layer
                break

    def forward(self, x):
        return self.model(x)
    
    def loss_function(self, prediction, y_true, warm_up=None, epoch=None, max_epochs=None, l1_lambda=0.0) -> Variable:
        if isinstance(prediction, tuple):
            prediction, _ = prediction
        
        if self.loss_type == "classic":
            base_loss = self._classic_multiclass(prediction, y_true)
        elif self.loss_type == "classic_binary":
            base_loss = self._classic_binary(prediction, y_true)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Add L1 penalty (only on the first layer)
        if l1_lambda > 0.0 and self.first_layer is not None:
            l1_penalty = torch.norm(self.first_layer.weight, p=1)
            base_loss += l1_lambda * l1_penalty

        return base_loss

    def _classic_multiclass(self, prediction, y_true):
        """
        Standard CrossEntropyLoss (classic classification).
        """
        y_true = y_true.to(prediction.device).long()
        criterion = nn.CrossEntropyLoss(weight=self.class_weights) 
        prediction_loss = criterion(prediction, y_true)
        return prediction_loss
    
    def _classic_binary(self, prediction, y_true):
        """
        Standard Binary CrossEntropyLoss (classic classification).
        """
        prediction = prediction.view(-1)
        # print("Prediction shape in class binary", prediction.shape)
        y_true = y_true.view(-1).float()
        # print("Y true shape in class binary", y_true.shape)
        # before: y_true = y_true.unsqueeze(1).float()
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        prediction_loss = criterion(prediction, y_true)
        return prediction_loss
    
    def set_hyperparameters(self, dropout_rate=None, l1_lambda=0.0, use_batchnorm=None):
        if dropout_rate is not None:
            self.dropout_rate = dropout_rate
            for layer in self.model:
                if isinstance(layer, nn.Dropout):
                    layer.p = dropout_rate

        if l1_lambda != 0.0:
            self.l1_lambda = l1_lambda 

        if use_batchnorm is not None:
            self.use_batchnorm = use_batchnorm
                
    def initialize_weights(self):
        for i, layer in enumerate(self.model):
            if isinstance(layer, nn.Linear):
                # Check if this is the last Linear layer
                is_last_layer = i + 1 < len(self.model) and not isinstance(self.model[i + 1], nn.Linear)
                if is_last_layer:
                    # Use Xavier initialization for the output layer
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                else:
                    # Use Kaiming initialization for hidden layers
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu')
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)