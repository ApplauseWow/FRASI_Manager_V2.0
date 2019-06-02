# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'model_estimate.ui'
#
# Created by: PyQt5 UI code generator 5.4.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Model_Estimate_UI(object):
    def setupUi(self, Model_Estimate_UI):
        Model_Estimate_UI.setObjectName("Model_Estimate_UI")
        Model_Estimate_UI.resize(810, 603)
        self.canvas = QtWidgets.QGraphicsView(Model_Estimate_UI)
        self.canvas.setGeometry(QtCore.QRect(70, 70, 671, 461))
        self.canvas.setObjectName("canvas")

        self.retranslateUi(Model_Estimate_UI)
        QtCore.QMetaObject.connectSlotsByName(Model_Estimate_UI)

    def retranslateUi(self, Model_Estimate_UI):
        _translate = QtCore.QCoreApplication.translate
        Model_Estimate_UI.setWindowTitle(_translate("Model_Estimate_UI", "Dialog"))

