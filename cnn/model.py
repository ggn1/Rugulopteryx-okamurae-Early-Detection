# TITLE: CNN For R. Okamurae detection.
# DESCRIPTION: This module defines a Convolutional Neural Network (CNN) model 
#              for detection of Rugulopteryx okamurae, a type of brown algae
#              native to Japan and Korea that has become inavasive in the
#              Mediterranean based on satellite images.

# IMPORTS
import os
import torch

class RoCNN(torch.nn.Module):
    """ Convolutional Neural Network for R. Okamurae detection. """

    def __init__(self):
        """ Constructor. 
        
        Arguments:
        h {int} -- Height of input images.
        w {int} -- Width of input images.
        c {int} -- Number of channels in input images.
        """
        super().__init__()

        # LAYERS
        # Convolutional Layers.
        self.cv1 = torch.nn.Sequential(
            torch.nn.Conv2d( # 64 X 64 X 6 => 31 X 31 X 32
                in_channels=6, out_channels=32, 
                padding=0, kernel_size=4, stride=(2, 2), 
                dtype=torch.float32, bias=True
            ), 
            torch.nn.ReLU(inplace=True)
        )
        self.cv2 = torch.nn.Sequential(
            torch.nn.Conv2d( # 15 X 15 X 32 => 15 X 15 X 64
                in_channels=32, out_channels=64, 
                padding=1, kernel_size=3, stride=(1, 1), 
                dtype=torch.float32, bias=True
            ),
            torch.nn.ReLU(inplace=True)
        )
        self.cv3 = torch.nn.Sequential(
            torch.nn.Conv2d( # 15 X 15 X 64 => 15 X 15 X 64
                in_channels=64, out_channels=64, 
                padding=1, kernel_size=3, stride=(1, 1), 
                dtype=torch.float32, bias=True
            ),
            torch.nn.ReLU(inplace=True)
        )
        self.cv4 = torch.nn.Sequential(
            torch.nn.Conv2d( # 15 X 15 X 64 => 15 X 15 X 64
                in_channels=64, out_channels=64, 
                padding=1, kernel_size=3, stride=(1, 1), 
                dtype=torch.float32, bias=True
            ),
            torch.nn.ReLU(inplace=True)
        )

        # Max Pooling Layers.
        # 31 X 31 X 32 => 15 X 15 X 32
        self.mp1 = torch.nn.MaxPool2d(kernel_size=3, stride=(2,2), padding=0)
        # 15 X 15 X 32 => 7 X 7 X 64
        self.mp2 = torch.nn.MaxPool2d(kernel_size=3, stride=(2,2), padding=0) 

        # Fully Connected Layers.
        self.fc1 =  torch.nn.Sequential( # 3136 => 512
            torch.nn.Linear(3136, 512, dtype=torch.float32, bias=True),
            torch.nn.ReLU(inplace=True)
        )
        self.fc2 =  torch.nn.Sequential( # 512 => 64
            torch.nn.Linear(512, 64, dtype=torch.float32, bias=True),
            torch.nn.ReLU(inplace=True)
        )
        self.fc3 =  torch.nn.Sequential( # 64 => 1
            torch.nn.Linear(64, 1, dtype=torch.float32, bias=True),
            torch.nn.Sigmoid()
        )

        # Flattening layers.
        self.flatten = torch.nn.Flatten()
        
    def forward(self, x):
        """ Forward pass.
        
        Arguments:
        x {torch.Tensor} -- Input tensor of shape (B, C, H, W).
        
        Returns:
        torch.Tensor -- Output tensor of shape (B, 1).
        """
        # Convolutional Layers with ReLU activations.
        
        # x [B, 6, 64, 64]
        
        x = self.cv1(x) # [B, 32, 31, 31]
        x = self.mp1(x) # [B, 32, 15, 15]
        x = self.cv2(x) # [B, 64, 15, 15]
        x = self.cv3(x) # [B, 64, 15, 15]
        x = self.cv4(x) # [B, 64, 15, 15]
        x = self.mp2(x) # [B, 64, 7, 7]

        # Flatten the tensor for Fully Connected Layers.
        x = self.flatten(x) # [B, 3136]

        # Fully Connected Layers with ReLU activations.
        x = self.fc1(x) # [B, 512]
        x = self.fc2(x) # [B, 64]
        x = self.fc3(x) # [B, 1]
        
        return x
    
def save_model(model, path, overwrite=False):
    ''' Saves model to given path. 
        @param model: The model to save.
        @param path: The path at which to save the model.
        @param overwrite: True if existing model at path is to be
                          overwritten. False otherwise. (Default = False)
    '''
    
    # if folder does not exist, create new one.
    folder_path = path[:path.rindex("\\")]
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print('created new folder at "{}"'.format(folder_path))
    elif os.path.exists(path):
        if not overwrite: return
            
    # save model
    torch.save(model.state_dict(keep_vars=True), path)
    print('model saved to "{}".'.format(path))

def load_model(path):
    ''' Load and return model at stored at a particular path.
        If no model was found at the given path, a new model
        may be created.
        @param path: The path from where the model is to be loaded.
                     Valid paths would be of the form 
                     "SAVED_MODELS_PATH\\mode\\name\\name_epoch_i.pth"
        @param h {int} -- Height of input images.
        @param w {int} -- Width of input images.
        @param c {int} -- Number of channels in input images.
        @return: Loaded/created model.
    '''
    create = False
    if not os.path.exists(path):
        input_value = input('model not found at path "{}", create new one? (y/n) '.format(path))
        if input_value in ["y", "Y"]: create = True

    # load or create model
    model = RoCNN()
    if create: print('new model created')
    else:
        model.load_state_dict(torch.load(path))
        print('model loaded from "{}"'.format(path))
    
    return model