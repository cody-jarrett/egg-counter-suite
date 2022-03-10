# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'egg_ana_gui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(261, 236)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.layoutWidget_3 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget_3.setGeometry(QtCore.QRect(10, 50, 241, 50))
        self.layoutWidget_3.setObjectName("layoutWidget_3")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget_3)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.selectedExperimentStaticLabel = QtWidgets.QLabel(self.layoutWidget_3)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.selectedExperimentStaticLabel.setFont(font)
        self.selectedExperimentStaticLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.selectedExperimentStaticLabel.setObjectName("selectedExperimentStaticLabel")
        self.verticalLayout.addWidget(self.selectedExperimentStaticLabel)
        self.selectedExperimentLabel = QtWidgets.QLabel(self.layoutWidget_3)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.selectedExperimentLabel.setFont(font)
        self.selectedExperimentLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.selectedExperimentLabel.setObjectName("selectedExperimentLabel")
        self.verticalLayout.addWidget(self.selectedExperimentLabel)
        self.selectExperimentButton = QtWidgets.QPushButton(self.centralwidget)
        self.selectExperimentButton.setGeometry(QtCore.QRect(60, 10, 141, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.selectExperimentButton.setFont(font)
        self.selectExperimentButton.setObjectName("selectExperimentButton")
        self.anaButton = QtWidgets.QPushButton(self.centralwidget)
        self.anaButton.setGeometry(QtCore.QRect(60, 170, 141, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.anaButton.setFont(font)
        self.anaButton.setObjectName("anaButton")
        self.minEggsLineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.minEggsLineEdit.setGeometry(QtCore.QRect(80, 130, 101, 20))
        self.minEggsLineEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.minEggsLineEdit.setObjectName("minEggsLineEdit")
        self.minEggsLabel = QtWidgets.QLabel(self.centralwidget)
        self.minEggsLabel.setGeometry(QtCore.QRect(40, 110, 181, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.minEggsLabel.setFont(font)
        self.minEggsLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.minEggsLabel.setObjectName("minEggsLabel")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 261, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.selectedExperimentStaticLabel.setText(_translate("MainWindow", "Selected Experiment"))
        self.selectedExperimentLabel.setText(_translate("MainWindow", "<ExperimentName>"))
        self.selectExperimentButton.setText(_translate("MainWindow", "Select Experiment"))
        self.anaButton.setText(_translate("MainWindow", "Analyze"))
        self.minEggsLabel.setText(_translate("MainWindow", "Minimum Eggs Threshold"))

