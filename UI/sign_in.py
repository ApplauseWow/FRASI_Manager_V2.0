# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'sign_in.ui'
#
# Created by: PyQt5 UI code generator 5.4.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Sign_In_UI(object):
    def setupUi(self, Sign_In_UI):
        Sign_In_UI.setObjectName("Sign_In_UI")
        Sign_In_UI.resize(696, 543)
        Sign_In_UI.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.ok = QtWidgets.QPushButton(Sign_In_UI)
        self.ok.setGeometry(QtCore.QRect(530, 490, 99, 27))
        self.ok.setObjectName("ok")
        self.again = QtWidgets.QPushButton(Sign_In_UI)
        self.again.setGeometry(QtCore.QRect(410, 490, 99, 27))
        self.again.setObjectName("again")
        self.scrollArea = QtWidgets.QScrollArea(Sign_In_UI)
        self.scrollArea.setGeometry(QtCore.QRect(80, 50, 511, 371))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 509, 369))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.gridLayoutWidget = QtWidgets.QWidget(self.scrollAreaWidgetContents)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(30, 30, 441, 281))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.table = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.table.setObjectName("table")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.retranslateUi(Sign_In_UI)
        QtCore.QMetaObject.connectSlotsByName(Sign_In_UI)

    def retranslateUi(self, Sign_In_UI):
        _translate = QtCore.QCoreApplication.translate
        Sign_In_UI.setWindowTitle(_translate("Sign_In_UI", "Sign In"))
        self.ok.setText(_translate("Sign_In_UI", "OK"))
        self.again.setText(_translate("Sign_In_UI", "Try again"))

