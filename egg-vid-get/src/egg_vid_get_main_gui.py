"""
User Interface layout file for the main GUI of the egg-vid-get software

Author: Cody Jarrett
Organization: Phillips Lab, Institute of Ecology and Evolution,
              University of Oregon
"""
# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'egg_vid_get_main_gui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1487, 771)
        font = QtGui.QFont()
        font.setPointSize(10)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.cameraSettingsGroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.cameraSettingsGroupBox.setGeometry(
            QtCore.QRect(10, 110, 180, 260)
        )
        font = QtGui.QFont()
        font.setPointSize(10)
        self.cameraSettingsGroupBox.setFont(font)
        self.cameraSettingsGroupBox.setObjectName("cameraSettingsGroupBox")
        self.brightnessSlider = QtWidgets.QSlider(self.cameraSettingsGroupBox)
        self.brightnessSlider.setGeometry(QtCore.QRect(10, 40, 161, 16))
        self.brightnessSlider.setOrientation(QtCore.Qt.Horizontal)
        self.brightnessSlider.setObjectName("brightnessSlider")
        self.contrastSlider = QtWidgets.QSlider(self.cameraSettingsGroupBox)
        self.contrastSlider.setGeometry(QtCore.QRect(10, 80, 161, 16))
        self.contrastSlider.setOrientation(QtCore.Qt.Horizontal)
        self.contrastSlider.setObjectName("contrastSlider")
        self.gammaSlider = QtWidgets.QSlider(self.cameraSettingsGroupBox)
        self.gammaSlider.setGeometry(QtCore.QRect(10, 120, 161, 16))
        self.gammaSlider.setOrientation(QtCore.Qt.Horizontal)
        self.gammaSlider.setObjectName("gammaSlider")
        self.brightnessSplitter = QtWidgets.QSplitter(
            self.cameraSettingsGroupBox
        )
        self.brightnessSplitter.setGeometry(QtCore.QRect(10, 20, 161, 21))
        self.brightnessSplitter.setOrientation(QtCore.Qt.Horizontal)
        self.brightnessSplitter.setObjectName("brightnessSplitter")
        self.brightnessLabel = QtWidgets.QLabel(self.brightnessSplitter)
        self.brightnessLabel.setObjectName("brightnessLabel")
        self.brightnessValLabel = QtWidgets.QLabel(self.brightnessSplitter)
        self.brightnessValLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.brightnessValLabel.setObjectName("brightnessValLabel")
        self.sharpnessSplitter = QtWidgets.QSplitter(
            self.cameraSettingsGroupBox
        )
        self.sharpnessSplitter.setGeometry(QtCore.QRect(10, 140, 161, 21))
        self.sharpnessSplitter.setOrientation(QtCore.Qt.Horizontal)
        self.sharpnessSplitter.setObjectName("sharpnessSplitter")
        self.sharpnessLabel = QtWidgets.QLabel(self.sharpnessSplitter)
        self.sharpnessLabel.setObjectName("sharpnessLabel")
        self.sharpnessValLabel = QtWidgets.QLabel(self.sharpnessSplitter)
        self.sharpnessValLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.sharpnessValLabel.setObjectName("sharpnessValLabel")
        self.sharpnessSlider = QtWidgets.QSlider(self.cameraSettingsGroupBox)
        self.sharpnessSlider.setGeometry(QtCore.QRect(10, 160, 161, 16))
        self.sharpnessSlider.setOrientation(QtCore.Qt.Horizontal)
        self.sharpnessSlider.setObjectName("sharpnessSlider")
        self.exposureSplitter = QtWidgets.QSplitter(
            self.cameraSettingsGroupBox
        )
        self.exposureSplitter.setGeometry(QtCore.QRect(10, 210, 161, 21))
        self.exposureSplitter.setOrientation(QtCore.Qt.Horizontal)
        self.exposureSplitter.setObjectName("exposureSplitter")
        self.exposureLabel = QtWidgets.QLabel(self.exposureSplitter)
        self.exposureLabel.setObjectName("exposureLabel")
        self.exposureValLabel = QtWidgets.QLabel(self.exposureSplitter)
        self.exposureValLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.exposureValLabel.setObjectName("exposureValLabel")
        self.exposureSlider = QtWidgets.QSlider(self.cameraSettingsGroupBox)
        self.exposureSlider.setGeometry(QtCore.QRect(10, 230, 161, 16))
        self.exposureSlider.setOrientation(QtCore.Qt.Horizontal)
        self.exposureSlider.setObjectName("exposureSlider")
        self.autoExposureLabel = QtWidgets.QLabel(self.cameraSettingsGroupBox)
        self.autoExposureLabel.setGeometry(QtCore.QRect(10, 180, 101, 21))
        self.autoExposureLabel.setObjectName("autoExposureLabel")
        self.autoExposureComboBox = QtWidgets.QComboBox(
            self.cameraSettingsGroupBox
        )
        self.autoExposureComboBox.setGeometry(QtCore.QRect(110, 180, 60, 20))
        self.autoExposureComboBox.setObjectName("autoExposureComboBox")
        self.contrastSplitter = QtWidgets.QSplitter(
            self.cameraSettingsGroupBox
        )
        self.contrastSplitter.setGeometry(QtCore.QRect(10, 60, 161, 21))
        self.contrastSplitter.setOrientation(QtCore.Qt.Horizontal)
        self.contrastSplitter.setObjectName("contrastSplitter")
        self.contrastLabel = QtWidgets.QLabel(self.contrastSplitter)
        self.contrastLabel.setObjectName("contrastLabel")
        self.contrastValLabel = QtWidgets.QLabel(self.contrastSplitter)
        self.contrastValLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.contrastValLabel.setObjectName("contrastValLabel")
        self.gammaSplitter = QtWidgets.QSplitter(self.cameraSettingsGroupBox)
        self.gammaSplitter.setGeometry(QtCore.QRect(10, 100, 161, 21))
        self.gammaSplitter.setOrientation(QtCore.Qt.Horizontal)
        self.gammaSplitter.setObjectName("gammaSplitter")
        self.gammaLabel = QtWidgets.QLabel(self.gammaSplitter)
        self.gammaLabel.setObjectName("gammaLabel")
        self.gammaValLabel = QtWidgets.QLabel(self.gammaSplitter)
        self.gammaValLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.gammaValLabel.setObjectName("gammaValLabel")
        self.startExperimentButton = QtWidgets.QPushButton(self.centralwidget)
        self.startExperimentButton.setGeometry(QtCore.QRect(10, 390, 180, 25))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.startExperimentButton.setFont(font)
        self.startExperimentButton.setObjectName("startExperimentButton")
        self.setMetadataButton = QtWidgets.QPushButton(self.centralwidget)
        self.setMetadataButton.setGeometry(QtCore.QRect(10, 10, 180, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.setMetadataButton.setFont(font)
        self.setMetadataButton.setObjectName("setMetadataButton")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 440, 181, 20))
        self.layoutWidget.setObjectName("layoutWidget")
        self.startHorizLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.startHorizLayout.setContentsMargins(0, 0, 0, 0)
        self.startHorizLayout.setObjectName("startHorizLayout")
        self.startLabel = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.startLabel.setFont(font)
        self.startLabel.setAlignment(
            QtCore.Qt.AlignLeading
            | QtCore.Qt.AlignLeft
            | QtCore.Qt.AlignVCenter
        )
        self.startLabel.setObjectName("startLabel")
        self.startHorizLayout.addWidget(self.startLabel)
        self.startDisplayLabel = QtWidgets.QLabel(self.layoutWidget)
        self.startDisplayLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.startDisplayLabel.setObjectName("startDisplayLabel")
        self.startHorizLayout.addWidget(self.startDisplayLabel)
        self.layoutWidget1 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget1.setGeometry(QtCore.QRect(10, 470, 181, 19))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.endHorizLayout = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.endHorizLayout.setContentsMargins(0, 0, 0, 0)
        self.endHorizLayout.setObjectName("endHorizLayout")
        self.endLabel = QtWidgets.QLabel(self.layoutWidget1)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.endLabel.setFont(font)
        self.endLabel.setAlignment(
            QtCore.Qt.AlignLeading
            | QtCore.Qt.AlignLeft
            | QtCore.Qt.AlignVCenter
        )
        self.endLabel.setObjectName("endLabel")
        self.endHorizLayout.addWidget(self.endLabel)
        self.endDisplayLabel = QtWidgets.QLabel(self.layoutWidget1)
        self.endDisplayLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.endDisplayLabel.setObjectName("endDisplayLabel")
        self.endHorizLayout.addWidget(self.endDisplayLabel)
        self.runtimeLabel = QtWidgets.QLabel(self.centralwidget)
        self.runtimeLabel.setGeometry(QtCore.QRect(60, 50, 80, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.runtimeLabel.setFont(font)
        self.runtimeLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.runtimeLabel.setObjectName("runtimeLabel")
        self.runtimeComboBox = QtWidgets.QComboBox(self.centralwidget)
        self.runtimeComboBox.setGeometry(QtCore.QRect(40, 70, 120, 20))
        self.runtimeComboBox.setObjectName("runtimeComboBox")
        self.frameDisplayLabel = QtWidgets.QLabel(self.centralwidget)
        self.frameDisplayLabel.setGeometry(QtCore.QRect(200, 10, 1280, 720))
        self.frameDisplayLabel.setAutoFillBackground(False)
        self.frameDisplayLabel.setStyleSheet("border: 1px solid black")
        self.frameDisplayLabel.setLineWidth(1)
        self.frameDisplayLabel.setText("")
        self.frameDisplayLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.frameDisplayLabel.setObjectName("frameDisplayLabel")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1487, 21))
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
        self.cameraSettingsGroupBox.setTitle(
            _translate("MainWindow", "Camera Settings")
        )
        self.brightnessLabel.setText(_translate("MainWindow", "Brightness"))
        self.brightnessValLabel.setText(_translate("MainWindow", "<Val>"))
        self.sharpnessLabel.setText(_translate("MainWindow", "Sharpness"))
        self.sharpnessValLabel.setText(_translate("MainWindow", "<Val>"))
        self.exposureLabel.setText(_translate("MainWindow", "Exposure"))
        self.exposureValLabel.setText(_translate("MainWindow", "<Val>"))
        self.autoExposureLabel.setText(
            _translate("MainWindow", "Auto Exposure")
        )
        self.contrastLabel.setText(_translate("MainWindow", "Contrast"))
        self.contrastValLabel.setText(_translate("MainWindow", "<Val>"))
        self.gammaLabel.setText(_translate("MainWindow", "Gamma"))
        self.gammaValLabel.setText(_translate("MainWindow", "<Val>"))
        self.startExperimentButton.setText(
            _translate("MainWindow", "Start Experiment")
        )
        self.setMetadataButton.setText(
            _translate("MainWindow", "Set Metadata")
        )
        self.startLabel.setText(_translate("MainWindow", "Start:"))
        self.startDisplayLabel.setText(_translate("MainWindow", "N/A"))
        self.endLabel.setText(_translate("MainWindow", "End:"))
        self.endDisplayLabel.setText(_translate("MainWindow", "N/A"))
        self.runtimeLabel.setText(_translate("MainWindow", "Runtime"))
