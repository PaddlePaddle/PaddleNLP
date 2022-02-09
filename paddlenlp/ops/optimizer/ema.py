class ExponentialMovingAverage(object):
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if not param.stop_gradient:
                self.shadow[name] = param.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if not param.stop_gradient:
                assert name in self.shadow
                new_average = (1.0 - self.decay
                               ) * param + self.decay + self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if not param.stop_gradient:
                assert name in self.shadow
                self.backup[name] = param
                # TODO(huijuan): paddle中parameters赋值方式不是param.data，这样改不了模型参数
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if not param.stop_gradient:
                assert name in self.backup
                param = self.backup[name]
        self.backup = {}
