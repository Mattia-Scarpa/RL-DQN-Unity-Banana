import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Agent - Policy Model.""" 

    def __init__(self, state_size, action_size, seed, hidden_layers=[256, 256, 256, 256]):
        """Initialize parameters and build model.
        
        Params
        ======
            state_size (int): Dimension of each state.
            action_size (int): Dimension of each action.
            seed (int): Random seed.
            hidden_layers (list): Number of nodes in hidden layers.
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        self.hidden_layers.extend([nn.Linear(hidden_layers[i], hidden_layers[i+1]) for i in range(len(hidden_layers)-1)])
        self.output_action = nn.Linear(hidden_layers[-1], action_size)
        self.output_value = nn.Linear(hidden_layers[-1], 1)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        for layer in self.hidden_layers:
            state = F.relu(layer(state))
        return self.output_action(state) + self.output_value(state)
    


#class QNetwork_visual(nn.Module):
#    """Agent - Policy Model.""" 
#
#    def __init__(self, state_size, action_size, seed, conv_hidden_layer = [256, 256, 256], hidden_layers=[512, 512, 512, 512, 512]):
#        """Initialize parameters and build model.
#        
#        Params
#        ======
#            state_size (int): Dimension of each state.
#            action_size (int): Dimension of each action.
#            seed (int): Random seed.
#            hidden_layers (list): Number of nodes in hidden layers.
#        """
#        super(QNetwork_visual, self).__init__()
#        self.seed = torch.manual_seed(seed)
#
#        self.pool = nn.MaxPool2d(2, 2)
#        #state_size[2]
#        self.conv_hidden_layer = nn.ModuleList([nn.Conv2d(1, conv_hidden_layer[0], kernel_size=3, stride=2, padding=1)])
#        self.conv_hidden_layer.extend([nn.Conv2d(conv_hidden_layer[i], conv_hidden_layer[i+1], kernel_size=3, stride=2, padding=1) for i in range(len(conv_hidden_layer)-1)])
#        
#        
#        self.hidden_layers = nn.ModuleList([nn.Linear(conv_hidden_layer[-1]*5*5, hidden_layers[0])])
#        self.hidden_layers.extend([nn.Linear(hidden_layers[i], hidden_layers[i+1]) for i in range(len(hidden_layers)-1)])
#        
#        self.output_action = nn.Linear(hidden_layers[-1], action_size)
#        self.output_value = nn.Linear(hidden_layers[-1], 1)
#
#    def forward(self, state):
#        """Build a network that maps state -> action values."""
#        for idx, layer in enumerate(self.conv_hidden_layer):
#            state = F.relu(layer(state))
#            if (idx + 1) % 3 == 0:  # execute pool after every 2 conv layers
#                state = self.pool(state)
#
#        print(state.size())
#
#        # flatten the state 
#        state = state.reshape(state.size(0), -1)
#
#        for layer in self.hidden_layers:
#            state = F.relu(layer(state))
#        return self.output_action(state) + self.output_value(state)
    