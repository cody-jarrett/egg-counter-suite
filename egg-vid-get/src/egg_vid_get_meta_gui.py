# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'egg_vid_get_meta_gui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(661, 608)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 150, 240, 81))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.setABacteriaLabel = QtWidgets.QLabel(self.groupBox)
        self.setABacteriaLabel.setGeometry(QtCore.QRect(10, 20, 60, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.setABacteriaLabel.setFont(font)
        self.setABacteriaLabel.setObjectName("setABacteriaLabel")
        self.setABacteriaLine = QtWidgets.QLineEdit(self.groupBox)
        self.setABacteriaLine.setGeometry(QtCore.QRect(70, 20, 160, 20))
        self.setABacteriaLine.setAlignment(QtCore.Qt.AlignCenter)
        self.setABacteriaLine.setObjectName("setABacteriaLine")
        self.setAWormsLabel = QtWidgets.QLabel(self.groupBox)
        self.setAWormsLabel.setGeometry(QtCore.QRect(10, 50, 60, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.setAWormsLabel.setFont(font)
        self.setAWormsLabel.setObjectName("setAWormsLabel")
        self.setAWormsLine = QtWidgets.QLineEdit(self.groupBox)
        self.setAWormsLine.setGeometry(QtCore.QRect(70, 50, 160, 20))
        self.setAWormsLine.setAlignment(QtCore.Qt.AlignCenter)
        self.setAWormsLine.setObjectName("setAWormsLine")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 260, 240, 81))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.setBBacteriaLabel = QtWidgets.QLabel(self.groupBox_2)
        self.setBBacteriaLabel.setGeometry(QtCore.QRect(10, 20, 60, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.setBBacteriaLabel.setFont(font)
        self.setBBacteriaLabel.setObjectName("setBBacteriaLabel")
        self.setBBacteriaLine = QtWidgets.QLineEdit(self.groupBox_2)
        self.setBBacteriaLine.setGeometry(QtCore.QRect(70, 20, 160, 20))
        self.setBBacteriaLine.setAlignment(QtCore.Qt.AlignCenter)
        self.setBBacteriaLine.setObjectName("setBBacteriaLine")
        self.setBWormsLabel = QtWidgets.QLabel(self.groupBox_2)
        self.setBWormsLabel.setGeometry(QtCore.QRect(10, 50, 60, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.setBWormsLabel.setFont(font)
        self.setBWormsLabel.setObjectName("setBWormsLabel")
        self.setBWormsLine = QtWidgets.QLineEdit(self.groupBox_2)
        self.setBWormsLine.setGeometry(QtCore.QRect(70, 50, 160, 20))
        self.setBWormsLine.setAlignment(QtCore.Qt.AlignCenter)
        self.setBWormsLine.setObjectName("setBWormsLine")
        self.setATableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.setATableWidget.setGeometry(QtCore.QRect(390, 30, 130, 510))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.setATableWidget.setFont(font)
        self.setATableWidget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setATableWidget.setObjectName("setATableWidget")
        self.setATableWidget.setColumnCount(1)
        self.setATableWidget.setRowCount(16)
        item = QtWidgets.QTableWidgetItem()
        self.setATableWidget.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.setATableWidget.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.setATableWidget.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.setATableWidget.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.setATableWidget.setVerticalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.setATableWidget.setVerticalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.setATableWidget.setVerticalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        self.setATableWidget.setVerticalHeaderItem(7, item)
        item = QtWidgets.QTableWidgetItem()
        self.setATableWidget.setVerticalHeaderItem(8, item)
        item = QtWidgets.QTableWidgetItem()
        self.setATableWidget.setVerticalHeaderItem(9, item)
        item = QtWidgets.QTableWidgetItem()
        self.setATableWidget.setVerticalHeaderItem(10, item)
        item = QtWidgets.QTableWidgetItem()
        self.setATableWidget.setVerticalHeaderItem(11, item)
        item = QtWidgets.QTableWidgetItem()
        self.setATableWidget.setVerticalHeaderItem(12, item)
        item = QtWidgets.QTableWidgetItem()
        self.setATableWidget.setVerticalHeaderItem(13, item)
        item = QtWidgets.QTableWidgetItem()
        self.setATableWidget.setVerticalHeaderItem(14, item)
        item = QtWidgets.QTableWidgetItem()
        self.setATableWidget.setVerticalHeaderItem(15, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(10)
        item.setFont(font)
        self.setATableWidget.setHorizontalHeaderItem(0, item)
        self.setBTableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.setBTableWidget.setGeometry(QtCore.QRect(520, 30, 130, 510))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.setBTableWidget.setFont(font)
        self.setBTableWidget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setBTableWidget.setObjectName("setBTableWidget")
        self.setBTableWidget.setColumnCount(1)
        self.setBTableWidget.setRowCount(16)
        item = QtWidgets.QTableWidgetItem()
        self.setBTableWidget.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.setBTableWidget.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.setBTableWidget.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.setBTableWidget.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.setBTableWidget.setVerticalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.setBTableWidget.setVerticalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.setBTableWidget.setVerticalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        self.setBTableWidget.setVerticalHeaderItem(7, item)
        item = QtWidgets.QTableWidgetItem()
        self.setBTableWidget.setVerticalHeaderItem(8, item)
        item = QtWidgets.QTableWidgetItem()
        self.setBTableWidget.setVerticalHeaderItem(9, item)
        item = QtWidgets.QTableWidgetItem()
        self.setBTableWidget.setVerticalHeaderItem(10, item)
        item = QtWidgets.QTableWidgetItem()
        self.setBTableWidget.setVerticalHeaderItem(11, item)
        item = QtWidgets.QTableWidgetItem()
        self.setBTableWidget.setVerticalHeaderItem(12, item)
        item = QtWidgets.QTableWidgetItem()
        self.setBTableWidget.setVerticalHeaderItem(13, item)
        item = QtWidgets.QTableWidgetItem()
        self.setBTableWidget.setVerticalHeaderItem(14, item)
        item = QtWidgets.QTableWidgetItem()
        self.setBTableWidget.setVerticalHeaderItem(15, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(10)
        item.setFont(font)
        self.setBTableWidget.setHorizontalHeaderItem(0, item)
        self.wormPositionLabel = QtWidgets.QLabel(self.centralwidget)
        self.wormPositionLabel.setGeometry(QtCore.QRect(470, 10, 100, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.wormPositionLabel.setFont(font)
        self.wormPositionLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.wormPositionLabel.setObjectName("wormPositionLabel")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(0, 0, 2, 2))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_14.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.layoutWidget1 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget1.setGeometry(QtCore.QRect(0, 0, 2, 2))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout_15.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.layoutWidget2 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget2.setGeometry(QtCore.QRect(0, 0, 2, 2))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout(self.layoutWidget2)
        self.horizontalLayout_16.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.layoutWidget3 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget3.setGeometry(QtCore.QRect(0, 0, 2, 2))
        self.layoutWidget3.setObjectName("layoutWidget3")
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout(self.layoutWidget3)
        self.horizontalLayout_17.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.saveMetadataButton = QtWidgets.QPushButton(self.centralwidget)
        self.saveMetadataButton.setGeometry(QtCore.QRect(250, 540, 120, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.saveMetadataButton.setFont(font)
        self.saveMetadataButton.setObjectName("saveMetadataButton")
        self.strainLine = QtWidgets.QLineEdit(self.centralwidget)
        self.strainLine.setGeometry(QtCore.QRect(100, 40, 140, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.strainLine.setFont(font)
        self.strainLine.setAlignment(QtCore.Qt.AlignCenter)
        self.strainLine.setObjectName("strainLine")
        self.strainLabel = QtWidgets.QLabel(self.centralwidget)
        self.strainLabel.setGeometry(QtCore.QRect(10, 40, 90, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.strainLabel.setFont(font)
        self.strainLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.strainLabel.setObjectName("strainLabel")
        self.chipLabel = QtWidgets.QLabel(self.centralwidget)
        self.chipLabel.setGeometry(QtCore.QRect(10, 70, 90, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.chipLabel.setFont(font)
        self.chipLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.chipLabel.setObjectName("chipLabel")
        self.chipLine = QtWidgets.QLineEdit(self.centralwidget)
        self.chipLine.setGeometry(QtCore.QRect(100, 70, 140, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.chipLine.setFont(font)
        self.chipLine.setAlignment(QtCore.Qt.AlignCenter)
        self.chipLine.setObjectName("chipLine")
        self.tempProfileBox = QtWidgets.QComboBox(self.centralwidget)
        self.tempProfileBox.setGeometry(QtCore.QRect(100, 10, 140, 20))
        self.tempProfileBox.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.tempProfileBox.setFont(font)
        self.tempProfileBox.setObjectName("tempProfileBox")
        self.tempProfileLabel = QtWidgets.QLabel(self.centralwidget)
        self.tempProfileLabel.setGeometry(QtCore.QRect(10, 10, 90, 20))
        self.tempProfileLabel.setBaseSize(QtCore.QSize(6, 0))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.tempProfileLabel.setFont(font)
        self.tempProfileLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.tempProfileLabel.setObjectName("tempProfileLabel")
        self.purposeLabel = QtWidgets.QLabel(self.centralwidget)
        self.purposeLabel.setGeometry(QtCore.QRect(10, 100, 90, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.purposeLabel.setFont(font)
        self.purposeLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.purposeLabel.setObjectName("purposeLabel")
        self.purposeLine = QtWidgets.QLineEdit(self.centralwidget)
        self.purposeLine.setGeometry(QtCore.QRect(100, 100, 280, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.purposeLine.setFont(font)
        self.purposeLine.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.purposeLine.setObjectName("purposeLine")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 661, 21))
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
        self.groupBox.setTitle(_translate("MainWindow", "Set A (1-16) Condition"))
        self.setABacteriaLabel.setText(_translate("MainWindow", "Bacteria:"))
        self.setAWormsLabel.setText(_translate("MainWindow", "Worms:"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Set B (17-32) Condition"))
        self.setBBacteriaLabel.setText(_translate("MainWindow", "Bacteria:"))
        self.setBWormsLabel.setText(_translate("MainWindow", "Worms:"))
        item = self.setATableWidget.verticalHeaderItem(0)
        item.setText(_translate("MainWindow", "1"))
        item = self.setATableWidget.verticalHeaderItem(1)
        item.setText(_translate("MainWindow", "2"))
        item = self.setATableWidget.verticalHeaderItem(2)
        item.setText(_translate("MainWindow", "3"))
        item = self.setATableWidget.verticalHeaderItem(3)
        item.setText(_translate("MainWindow", "4"))
        item = self.setATableWidget.verticalHeaderItem(4)
        item.setText(_translate("MainWindow", "5"))
        item = self.setATableWidget.verticalHeaderItem(5)
        item.setText(_translate("MainWindow", "6"))
        item = self.setATableWidget.verticalHeaderItem(6)
        item.setText(_translate("MainWindow", "7"))
        item = self.setATableWidget.verticalHeaderItem(7)
        item.setText(_translate("MainWindow", "8"))
        item = self.setATableWidget.verticalHeaderItem(8)
        item.setText(_translate("MainWindow", "9"))
        item = self.setATableWidget.verticalHeaderItem(9)
        item.setText(_translate("MainWindow", "10"))
        item = self.setATableWidget.verticalHeaderItem(10)
        item.setText(_translate("MainWindow", "11"))
        item = self.setATableWidget.verticalHeaderItem(11)
        item.setText(_translate("MainWindow", "12"))
        item = self.setATableWidget.verticalHeaderItem(12)
        item.setText(_translate("MainWindow", "13"))
        item = self.setATableWidget.verticalHeaderItem(13)
        item.setText(_translate("MainWindow", "14"))
        item = self.setATableWidget.verticalHeaderItem(14)
        item.setText(_translate("MainWindow", "15"))
        item = self.setATableWidget.verticalHeaderItem(15)
        item.setText(_translate("MainWindow", "16"))
        item = self.setATableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Set A"))
        item = self.setBTableWidget.verticalHeaderItem(0)
        item.setText(_translate("MainWindow", "17"))
        item = self.setBTableWidget.verticalHeaderItem(1)
        item.setText(_translate("MainWindow", "18"))
        item = self.setBTableWidget.verticalHeaderItem(2)
        item.setText(_translate("MainWindow", "19"))
        item = self.setBTableWidget.verticalHeaderItem(3)
        item.setText(_translate("MainWindow", "20"))
        item = self.setBTableWidget.verticalHeaderItem(4)
        item.setText(_translate("MainWindow", "21"))
        item = self.setBTableWidget.verticalHeaderItem(5)
        item.setText(_translate("MainWindow", "22"))
        item = self.setBTableWidget.verticalHeaderItem(6)
        item.setText(_translate("MainWindow", "23"))
        item = self.setBTableWidget.verticalHeaderItem(7)
        item.setText(_translate("MainWindow", "24"))
        item = self.setBTableWidget.verticalHeaderItem(8)
        item.setText(_translate("MainWindow", "25"))
        item = self.setBTableWidget.verticalHeaderItem(9)
        item.setText(_translate("MainWindow", "26"))
        item = self.setBTableWidget.verticalHeaderItem(10)
        item.setText(_translate("MainWindow", "27"))
        item = self.setBTableWidget.verticalHeaderItem(11)
        item.setText(_translate("MainWindow", "28"))
        item = self.setBTableWidget.verticalHeaderItem(12)
        item.setText(_translate("MainWindow", "29"))
        item = self.setBTableWidget.verticalHeaderItem(13)
        item.setText(_translate("MainWindow", "30"))
        item = self.setBTableWidget.verticalHeaderItem(14)
        item.setText(_translate("MainWindow", "31"))
        item = self.setBTableWidget.verticalHeaderItem(15)
        item.setText(_translate("MainWindow", "32"))
        item = self.setBTableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Set B"))
        self.wormPositionLabel.setText(_translate("MainWindow", "Worm Position"))
        self.saveMetadataButton.setText(_translate("MainWindow", "Save Metadata"))
        self.strainLabel.setText(_translate("MainWindow", "Strain:"))
        self.chipLabel.setText(_translate("MainWindow", "Chip #:"))
        self.tempProfileLabel.setText(_translate("MainWindow", "Temp Profile:"))
        self.purposeLabel.setText(_translate("MainWindow", "Purpose:"))
