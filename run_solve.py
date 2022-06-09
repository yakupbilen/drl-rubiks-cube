"""
For running: python run_solve.py -p "ini_files/solve.ini"
"""
import argparse
import sys
from widgets.cube_widget import Ui_MainWindow
from configs.solve_conf import SolveConfig
from PyQt5 import QtWidgets
import numpy as np
seq = np.random.randint(0, 12, 10)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune gamma")
    parser.add_argument("-p", "--ini_path", type=str, metavar="", required=True,
                        help="Path of config file. Extension of file must be .ini")
    args = parser.parse_args()

    conf = SolveConfig(args.ini_path)
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow, conf)
    MainWindow.show()
    sys.exit(app.exec_())








