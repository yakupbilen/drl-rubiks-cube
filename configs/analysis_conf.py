import logging
import configparser


class AnalysisConfig:
    log = logging.getLogger("AnalysisConfig")

    def __init__(self, file_name):
        self.data = configparser.ConfigParser()
        self.log.info("Reading config file %s", file_name)
        if not self.data.read(file_name):
            raise ValueError("Config file %s not found" % file_name)

    @property
    def sect_analysis(self):
        return self.data['analysis']

    @property
    def network_type(self):
        return self.sect_analysis["network_type"]

    @property
    def heuristic_path(self):
        return self.sect_analysis["heuristic_path"]

    @property
    def target_path(self):
        return self.sect_analysis["target_path"]

    @property
    def weight(self):
        return self.sect_analysis.getfloat("weight")

    @property
    def max_iter(self):
        return self.sect_analysis.getint('max_iter')

    @property
    def max_time(self):
        return self.sect_analysis.getint('max_time')

    @property
    def batch_size(self):
        return self.sect_analysis.getint('batch_size')

    @property
    def scramble_depth(self):
        return self.sect_analysis.getint('scramble_depth')

    @property
    def games(self):
        return self.sect_analysis.getint('games')


