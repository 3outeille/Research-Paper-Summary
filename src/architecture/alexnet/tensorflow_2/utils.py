import numpy as np

def load_weights(model, dic):
    for key in dic:
        if key in ['conv2', 'conv4', 'conv5']:
            half = dic[key][0].shape[-1] // 2
            model.get_layer(key + '_1').set_weights([dic[key][0][..., :half], dic[key][1][:half]])
            model.get_layer(key + '_2').set_weights([dic[key][0][..., half:], dic[key][1][half:]])
        else:
            model.get_layer(key).set_weights(dic[key])
    print('Loading complete.')
    return model


def check_loaded(model, dic):
     for key in dic:
        if key in ['conv2', 'conv4', 'conv5']:
            weights = np.concatenate([model.get_layer(key + '_1').get_weights()[0],
                                      model.get_layer(key + '_2').get_weights()[0]], axis=-1)

            biases = np.concatenate([model.get_layer(key + '_1').get_weights()[1],
                                     model.get_layer(key + '_2').get_weights()[1]], axis=-1)

            isWeights = (weights == dic[key][0]).sum()
            isBiases = (biases == dic[key][1]).sum()

        else:
            isWeights = (model.get_layer(key).get_weights()[0] == dic[key][0]).sum()
            isBiases = (model.get_layer(key).get_weights()[1] == dic[key][1]).sum()

        isWeightsLoaded = (isWeights == np.prod(dic[key][0].shape))
        isBiasesLoaded = (isBiases == dic[key][1].shape)
   
        if isWeightsLoaded == isBiasesLoaded:
            print('{}: weights -> Loaded | biases -> Loaded'.format(key))
        elif isWeightsLoaded:
            print('{}: weights -> Loaded | biases -> Not Loaded'.format(key))
        elif isBiasesLoaded:
            print('{}: weights -> Not Loaded | biases -> Loaded'.format(key))
        else: 
            print('{}: weights -> Not Loaded | biases -> Not Loaded'.format(key))