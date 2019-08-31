import numpy as np
import torch
import attr
import fastai.text
from allennlp.modules import elmo


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


class LMWrapper:

    def get_last_hiddens_batch(self):
        raise NotImplementedError()

    def get_batch_items(self):
        raise NotImplementedError()


@attr.s
class AllenELMoWrapper:
    elmo.Elmo
    elmo = attr.ib()
    tokenizer = attr.ib(default=lambda s: s.split())

    def get_last_hiddens_batch(self, texts):
        char_ids = self._get_char_ids(texts)
        self.elmo(char_ids)
        return self.elmo._elmo_lstm._elmo_lstm._states

    def get_batch_items(self, texts):
        elmo_output = self.elmo(self._get_char_ids(texts))
        return elmo_output['elmo_representations']

    def _get_char_ids(self, texts):
        tokens_lists = [self.tokenizer(t) for t in texts]
        return elmo.batch_to_ids(tokens_lists)


@attr.s
class FastAILearnerWrapper:

    learner = attr.ib()

    def get_batch_items(self, texts, cpu=False):
        return torch.cat([self.learner.data.one_item(text, cpu=cpu)[0] for text in texts])

    def get_model_outputs(self, text, cpu=False):
        input_tensor, __ = self.learner.data.one_item(text, cpu=cpu)
        return self.learner.model[0](input_tensor)

    def get_last_hiddens(self, text, layers=(0, 1, 2)):
        input_tensor, __ = self.learner.data.one_item(text)
        self.learner.model[0](input_tensor)
        hiddens = self.learner.model[0].hidden
        hiddens = [h.cpu().numpy().reshape(1, -1) for h in hiddens]
        hiddens = hiddens[::-1] # do this because LM layers are reversed
        hiddens = [hiddens[i] for i in layers]
        return np.hstack(hiddens)

    def get_last_hiddens_batch(self, texts, layers=(0, 1, 2)):
        return np.vstack([self.get_last_hiddens(self.learner, text, layers) for text in texts])

