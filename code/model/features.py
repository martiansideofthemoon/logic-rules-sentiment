import numpy as np


class NeuralNetwork(object):
    def __init__(self, args, instance, info):
        self.seq_len = instance['sentence_len']
        self.sentence = instance['sentence']
        self.model = info['model_eval']

        self.final_inputs = np.array(self.sentence)
        self.final_string = instance['pad_string']
        self.final_seq_len = np.array(self.seq_len)

    def postprocess_func(self, output):
        # A logarithm is taken here to account for softmax in logicnn.compute_probability
        self.output = np.log(output)
        return self.output


class LogicRuleBut(object):
    def __init__(self, args, instance, info):
        self.seq_len = instance['sentence_len']
        self.sentence = instance['sentence']
        self.model = info['model_eval']
        self.vocab = vocab = info['vocab']
        self.rev_vocab = rev_vocab = info['rev_vocab']

        self.hasBut = vocab['but'] in self.sentence
        if self.hasBut is True:
            first_but = self.sentence.index(vocab['but'])
            new_sent = self.sentence[first_but + 1:]
            self.new_sent = [vocab['<PAD>']] * 4 + new_sent + [vocab['<PAD>']] * (self.seq_len - len(new_sent) - 4)
            self.final_inputs = np.array(self.new_sent)
            self.final_string = " ".join([rev_vocab[x] for x in self.new_sent])
            self.final_seq_len = np.array(self.seq_len)
        else:
            self.final_string = instance['pad_string']
            self.final_inputs = np.array(self.sentence)
            self.final_seq_len = np.array(self.seq_len)
            # Storing a mask with all 0
            self.A_mask = np.zeros(self.seq_len)

    def postprocess_func(self, output):
        # We directly use the output probabilities as the rule value
        # We ignore the -1 since it won't affect computation
        # In Hu et al. 2016, C = 6 and \lambda = 1
        if self.hasBut is True:
            self.output = output
        else:
            # These will cancel off when the sentence doesn't have 'but'
            self.output = np.array([0.5, 0.5])
        return self.output


features = [
    NeuralNetwork,
    LogicRuleBut
]
