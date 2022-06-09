import logging
import configparser


class SolveConfig:
    log = logging.getLogger("SolveConfig")

    def __init__(self, file_name):
        self.data = configparser.ConfigParser()
        self.log.info("Reading config file %s", file_name)
        if not self.data.read(file_name):
            raise ValueError("Config file %s not found" % file_name)


    @property
    def sect_search(self):
        return self.data['search']

    @property
    def network_type(self):
        return self.sect_search["network_type"]

    @property
    def weight(self):
        return self.sect_search.getfloat("weight")

    @property
    def heuristic_path(self):
        return self.sect_search["heuristic_path"]

    @property
    def batch_size(self):
        return self.sect_search.getint('batch_size')