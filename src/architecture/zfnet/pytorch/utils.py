from collections import OrderedDict
import torch

def load_model_with_diff_keys(pretrained_model_file, target_model):
    """
        Loads a pretrained model parameters into our target model.
        We assume that there is only mistmatch between key names
        and tensor dimension are same.
        
        Parameters:
        -pretrained_model_file: .pth file.
        -target_model: model to load parameters.
        
    """
    pretrained_model = torch.load(pretrained_model_file)
    new_state_dict = OrderedDict()
    model_key = list(target_model.state_dict().keys())
    
    count = 0
    for key, value in pretrained_model.items():
        new_key = model_key[count]
        new_state_dict[new_key] = value
        count += 1
        
    target_model.load_state_dict(new_state_dict)
