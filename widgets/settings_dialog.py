from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIntValidator,QDoubleValidator
class Ui_SettingsDialog(QtWidgets.QDialog):
    def __init__(self,settings):
        self.settings = settings
        super(Ui_SettingsDialog, self).__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self,self.settings)


class Ui_Dialog(object):
    def setupUi(self, Dialog,settings):
        self.settings = settings
        Dialog.setObjectName("Dialog")
        Dialog.resize(500, 320)
        Dialog.setMinimumSize(QtCore.QSize(400, 320))
        Dialog.setMaximumSize(QtCore.QSize(900, 350))
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.formLayout_2 = QtWidgets.QFormLayout()
        self.formLayout_2.setObjectName("formLayout_2")
        self.label = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.line_delay = QtWidgets.QLineEdit(Dialog)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.line_delay.setFont(font)
        self.line_delay.setMaxLength(4)
        self.line_delay.setObjectName("line_delay")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.line_delay)
        self.label_2 = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.line_weight = QtWidgets.QLineEdit(Dialog)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.line_weight.setFont(font)
        self.line_weight.setMaxLength(4)
        self.line_weight.setObjectName("line_weight")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.line_weight)
        self.label_3 = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.line_batch_size = QtWidgets.QLineEdit(Dialog)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.line_batch_size.setFont(font)
        self.line_batch_size.setMaxLength(3)
        self.line_batch_size.setObjectName("line_batch_size")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.line_batch_size)
        self.label_4 = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.line_def_depth = QtWidgets.QLineEdit(Dialog)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.line_def_depth.setFont(font)
        self.line_def_depth.setMaxLength(3)
        self.line_def_depth.setObjectName("line_def_depth")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.line_def_depth)
        self.label_5 = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.formLayout_2.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.line_default_time_threshold = QtWidgets.QLineEdit(Dialog)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.line_default_time_threshold.setFont(font)
        self.line_default_time_threshold.setMaxLength(4)
        self.line_default_time_threshold.setObjectName("line_default_time_threshold")
        self.formLayout_2.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.line_default_time_threshold)
        self.label_6 = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.formLayout_2.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.line_default_iter_threshold = QtWidgets.QLineEdit(Dialog)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.line_default_iter_threshold.setFont(font)
        self.line_default_iter_threshold.setMaxLength(4)
        self.line_default_iter_threshold.setObjectName("line_default_iter_threshold")
        self.formLayout_2.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.line_default_iter_threshold)
        self.verticalLayout.addLayout(self.formLayout_2)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)
        self.verticalLayout.addWidget(self.buttonBox)
        self.line_delay.setValidator(QIntValidator())
        self.line_batch_size.setValidator(QIntValidator())
        self.line_default_iter_threshold.setValidator(QIntValidator())
        self.line_default_time_threshold.setValidator(QIntValidator())
        self.line_def_depth.setValidator(QIntValidator())
        self.line_weight.setValidator(QDoubleValidator(0.0, 1.0, 2))

        self.line_delay.setPlaceholderText(f"Old value : {self.settings.actions_delay}ms")
        self.line_weight.setPlaceholderText(f"Old value : {self.settings.search_weight}")
        self.line_default_time_threshold.setPlaceholderText(f"Old value : {self.settings.def_time}s")
        self.line_default_iter_threshold.setPlaceholderText(f"Old value : {self.settings.def_time}")
        self.line_batch_size.setPlaceholderText(f"Old value : {self.settings.search_batch_size}")
        self.line_def_depth.setPlaceholderText(f"Old value : {self.settings.def_scramble_depth}")

        self.buttonBox.accepted.connect(self.accept)
        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def validate(self,value,lower,upper=None):
        if value<lower:
            return False
        if upper:
            if value>upper:
                return False
        return True


    def accept(self):
        delay = self.line_delay.text() or self.settings.actions_delay
        delay = int(delay)
        if not self.validate(delay,0):
            delay = self.settings.actions_delay
        self.settings.actions_delay = delay
        weight = self.line_weight.text() or self.settings.search_weight
        weight = float(weight)
        if not self.validate(weight,0,1):
            weight = self.settings.search_weight
        self.settings.search_weight = weight
        def_time = self.line_default_time_threshold.text() or self.settings.def_time
        def_time = int(def_time)
        if not self.validate(def_time,1):
            def_time = self.settings.def_time
        self.settings.def_time = def_time
        def_iter = self.line_default_iter_threshold.text() or self.settings.def_iter
        def_iter = int(def_iter)
        if not self.validate(def_iter,1):
            def_iter = self.settings.def_iter
        self.settings.def_iter = def_iter
        batch_size = self.line_batch_size.text() or self.settings.search_batch_size
        batch_size = int(batch_size)
        if not self.validate(batch_size,1,1000):
            batch_size = self.settings.search_batch_size
        self.settings.search_batch_size = batch_size
        def_depth = self.line_def_depth.text() or self.settings.def_scramble_depth
        def_depth = int(def_depth)
        if not self.validate(def_depth,1,200):
            def_depth = self.settings.def_scramble_depth
        self.settings.def_scramble_depth = def_depth

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "Actions Delay(ms.) :"))
        self.label_2.setText(_translate("Dialog", "A* weight :"))
        self.label_3.setText(_translate("Dialog", "A* Batch Size :"))
        self.label_4.setText(_translate("Dialog", "Default scramble depth : "))
        self.label_5.setText(_translate("Dialog", "Default time threshold :"))
        self.label_6.setText(_translate("Dialog", "Default iter. threshold : "))

