import logging
import configparser


class TrainConfig:
    log = logging.getLogger("TrainConfig")

    def __init__(self, file_name):
        self.data = configparser.ConfigParser()
        self.log.info("Reading config file %s", file_name)
        if not self.data.read(file_name):
            raise ValueError("Config file %s not found" % file_name)


    @property
    def sect_train(self):
        return self.data['train']

    @property
    def sect_general(self):
        return self.data['general']

    @property
    def network_type(self):
        return self.sect_general['network_type']

    @property
    def device(self):
        return self.sect_general["device"]

    @property
    def include_first(self):
        return self.sect_train.getboolean('include_first')

    @property
    def learning_rate(self):
        return self.sect_train.getfloat('lr')

    @property
    def max_iter(self):
        return self.sect_train.getint('max_iter')

    @property
    def batch_size(self):
        return self.sect_train.getint('batch_size')

    @property
    def scramble_depth(self):
        return self.sect_train.getint('scramble_depth')

    @property
    def report_batches(self):
        return self.sect_train.getint('report_batches')

    @property
    def lr_decay_gamma(self):
        return self.sect_train.getfloat('lr_decay_gamma', fallback=1.0)

    @property
    def lr_decay_interval(self):
        return self.sect_train.getint('lr_decay_interval')

    @property
    def weight_type(self):
        return self.sect_train.get('weight_type')

    @property
    def tau(self):
        return self.sect_train.getfloat("tau")

    @property
    def generator_interval(self):
        return self.sect_train.getint("generator_interval")

    @property
    def checkpoint(self):
        return self.sect_train.getint("checkpoint")