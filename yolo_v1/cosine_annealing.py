import math
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K


class CosineAnnealingScheduler(Callback):
    """Cosine annealing scheduler.
    """

    def __init__(self, eta_max, eta_min=0, T_max=10, T_mult=2, decay=1, verbose=0):
        super(CosineAnnealingScheduler, self).__init__()
        self.T_max = T_max
        self.T_mult = T_mult
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose
        self.prev_epochs = 0
        self.bool = False
        self.decay = decay

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        e = epoch - self.prev_epochs
        theta = math.pi * e / self.T_max

        if round(math.cos(theta), 6) < 1e-7 and self.bool == False:
            self.bool = True

        if round(math.cos(theta), 3) > 0.9999 and self.bool:
            self.eta_max = self.eta_max * self.decay
            self.bool = False
        # lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(theta)) / 2     #default
        lr = self.eta_min + (self.eta_max - self.eta_min) * (3 + math.cos(theta)) / 4

        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealingScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))
        if e != 0 and e % self.T_max == 0:
            self.prev_epochs = epoch + 1
            self.T_max *= self.T_mult

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
