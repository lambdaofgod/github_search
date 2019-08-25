import numpy as np
import torch


def print_shapes_recursively(tpl, nesting=''):
    if type(tpl) is tuple or type(tpl) is list:
        l = len(tpl)
        print(nesting + 'Collection of {} elements:'.format(l))
        for item in tpl:
            print_shapes_recursively(item, nesting + '\t')
    else:
        print(nesting + str(tpl.shape))


def get_shapes_recursively(tpl, nesting=''):
    if type(tpl) is tuple or type(tpl) is list:
        l = len(tpl)
        print(nesting + 'Collection of {} elements:'.format(l))
        for item in tpl:
            print_shapes_recursively(item, nesting + '\t')
    else:
        print(nesting + str(tpl.shape))


def get_batch_items(learner, texts):
    return torch.cat([learner.data.one_item(text)[0] for text in texts])


def get_model_outputs(learner, text):
    input_tensor, __ = learner.data.one_item(text)
    return learner.model[0](input_tensor)


def get_last_hiddens(learner, text, layers=(0, 1, 2)):
    input_tensor, __ = learner.data.one_item(text)
    learner.model[0](input_tensor)
    hiddens = learner.model[0].hidden
    hiddens = [h.cpu().numpy().reshape(1, -1) for h in hiddens]
    hiddens = hiddens[::-1] # do this because LM layers are reversed
    hiddens = [hiddens[i] for i in layers]
    return np.hstack(hiddens)


def get_last_hiddens_batch(learner, texts, layers=(0, 1, 2)):
    return np.vstack([get_last_hiddens(learner, text, layers) for text in texts])