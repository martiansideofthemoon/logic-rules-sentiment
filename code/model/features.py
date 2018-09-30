import numpy as np


class Feature(object):
    needs_nn = False
    postprocess = False


class NeuralNetwork(Feature):
    needs_nn = True
    postprocess = True

    def __init__(self, args, instance, info):
        self.seq_len = instance['sentence_len']
        self.sentence = instance['sentence']
        self.model = info['model_eval']

        self.final_inputs = np.array(self.sentence)
        self.final_string = instance['pad_string']
        self.final_seq_len = np.array(self.seq_len)

        self.use_log_prob = args.config.use_log_prob

    def postprocess_func(self, output):
        if self.use_log_prob is True:
            self.output = np.log(output)
        else:
            self.output = output
        return self.output


class ButIndicator(Feature):
    needs_nn = False
    postprocess = False

    def __init__(self, args, instance, info):
        vocab = info['vocab']
        self.sentence = instance['sentence']
        self.hasBut = vocab['but'] in self.sentence
        if self.hasBut is True:
            self.output = np.array([1.0, 1.0])
        else:
            self.output = np.array([0.0, 0.0])
        self.final_string = instance['pad_string']

    def generate(self):
        return self.output


class LogicRuleBut(Feature):
    needs_nn = True
    postprocess = True

    def __init__(self, args, instance, info):
        self.seq_len = instance['sentence_len']
        self.sentence = instance['sentence']
        self.model = info['model_eval']
        self.vocab = vocab = info['vocab']
        self.rev_vocab = rev_vocab = info['rev_vocab']

        self.use_log_prob = args.config.use_log_prob

        self.hasBut = vocab['but'] in self.sentence
        if self.hasBut is True:
            first_but = self.sentence.index(vocab['but'])
            new_sent = self.sentence[first_but + 1:]
            self.new_sent = [vocab['<PAD>']] * 4 + new_sent + [vocab['<PAD>']] * (self.seq_len - len(new_sent) - 4)
            self.final_inputs = np.array(self.new_sent)
            self.final_string = " ".join([rev_vocab[x] for x in self.new_sent])
            self.final_seq_len = np.array(self.seq_len)
            # Storing a mask with only A values = 1, rest 0
            self.A_mask = np.concatenate((
                np.ones(first_but),
                np.zeros(self.seq_len - first_but)
            ))
        else:
            self.final_string = instance['pad_string']
            self.final_inputs = np.array(self.sentence)
            self.final_seq_len = np.array(self.seq_len)
            # Storing a mask with all 0
            self.A_mask = np.zeros(self.seq_len)

    def postprocess_func(self, output):
        if self.hasBut is True:
            self.output = output
        else:
            self.output = np.array([0.5, 0.5])
        # In the case we wish to take `log` over these probabilities too, uncomment -
        # if self.use_log_prob is True:
        #     self.output = np.log(self.output)
        return self.output


class IndicatorTimesNN(Feature):
    needs_nn = False
    postprocess = False

    def __init__(self, args, instance, info):
        self.instance = instance

    def generate(self):
        self.output = np.multiply(
            self.instance['features'][0].output,
            self.instance['features'][2].output
        )
        return self.output


features = [
    NeuralNetwork,
    LogicRuleBut
]