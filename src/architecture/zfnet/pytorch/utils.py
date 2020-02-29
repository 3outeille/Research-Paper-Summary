from collections import OrderedDict
import torch

# def load_model_with_diff_keys(pretrained_model_file, target_model):
#     """
#         Loads a pretrained model parameters into our target model.
#         We assume that there is only mistmatch between key names
#         and tensor dimension are same.
        
#         Parameters:
#         -pretrained_model_file: .pth file.
#         -target_model: model to load parameters.
        
#     """
#     pretrained_model = torch.load(pretrained_model_file)
#     new_state_dict = OrderedDict()
#     model_key = list(target_model.state_dict().keys())
    
#     count = 0
#     for key, value in pretrained_model.items():
#         new_key = model_key[count]
#         new_state_dict[new_key] = value
#         count += 1
        
#     target_model.load_state_dict(new_state_dict)
    
def load_model(pretrained_model_file, target_model):
    """
        Loads a pretrained model parameters into our target model.
        where target model parameters names are different from the
        pretrained one.
        
        Load parameters to deconv part too.
        
        Parameters:
        -pretrained_model_file: .pth file.
        -target_model: model to load parameters.
        
    """
    pretrained_model = torch.load(pretrained_model_file)
    new_state_dict_1 = OrderedDict()
    model_key = list(target_model.state_dict().keys())
    

    count = 0
    for key, value in pretrained_model.items():
        new_key = model_key[count]
        new_state_dict_1[new_key] = value
        count += 1

    mapping = {'features.conv1.weight': 'deconv_conv1.weight',
               'features.conv2.weight': 'deconv_conv2.weight',
               'features.conv3.weight': 'deconv_conv3.weight',
               'features.conv4.weight': 'deconv_conv4.weight',
               'features.conv5.weight': 'deconv_conv5.weight'}
    
    new_state_dict_2 = OrderedDict()
    # Load Deconv part
    for key, value in new_state_dict_1.items():
        if key in mapping:
            new_state_dict_2[mapping[key]] = value
    
    
    target_model.load_state_dict({**new_state_dict_1, **new_state_dict_2})