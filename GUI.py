# -*-coding:utf-8-*-

from PyQt5.QtWidgets import QMainWindow, QDialog, QApplication
from PyQt5 import QtCore
from UI.index import Ui_Index_UI
from UI.identify_id import Ui_Identity_ID_UI
from UI.sign_in import Ui_Sign_In_UI
from UI.sys_option import Ui_sys_option
from UI.model_estimate import Ui_Model_Estimate_UI

# GUI classes


class Index(QMainWindow, Ui_Index_UI):
    """
    GUI -- main window
    """

    def __init__(self):
        super(Index, self).__init__()
        self.setupUi(self)
        self.menu.expandAll()
        # the group of widgets >>>
        self.info_list = (self.nameLabel, self.name,
                          self.iDLabel, self.id_num,
                          self.dateLabel, self.date,
                          self.status)
        # structure of ui >>>
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)  # move title
        # self.setAttribute(Qt.WA_TranslucentBackground) # transparent
        desktop = QApplication.desktop()
        desk_width = desktop.availableGeometry().width()
        desk_height = desktop.availableGeometry().height()
        self.resize(desk_width, desk_height)
        self.menu.expandAll()
        self.menu.setFixedWidth(desk_width // 8)
        self.frame.resize(desk_width - self.menu.width(), desk_height)
        self.show_hide.setFixedSize(self.menu.width() // 10, desk_height)
        self.show_hide.setText(">")
        # <<< structure of ui
        self.frame.setScaledContents(True)
        self.showFullScreen()


class Identify_Id_UI(QDialog, Ui_Identity_ID_UI):
    """
    GUI -- identify-id window
    """

    def __init__(self):
        super(Identify_Id_UI, self).__init__()
        self.setupUi(self)


class Sign_In_UI(QDialog, Ui_Sign_In_UI):
    """
    GUI -- sign-in window
    """

    def __init__(self):
        super(Sign_In_UI, self).__init__()
        self.setupUi(self)


class Sys_Option_UI(QDialog, Ui_sys_option):
    """
    GUI -- sys-option window
    """

    def __init__(self):
        super(Sys_Option_UI, self).__init__()
        self.setupUi(self)


class Model_Estimate_UI(QDialog, Ui_Model_Estimate_UI):
    """
    GUI -- model-estimate window
    """

    def __init__(self):
        super(Model_Estimate_UI, self).__init__()
        self.setupUi(self)



