# -*-coding:utf-8-*-
from PyQt5.QtWidgets import QMainWindow, QWidget, QApplication, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPixmap, QImage


class Index_UI(QMainWindow):
    """
    the static GUI of main window
    """

    def __init__(self):
        super(Index_UI, self).__init__()
        self.setWindowFlags(Qt.FramelessWindowHint) # no border
        # self.setAttribute(Qt.WA_TranslucentBackground) # transparent
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.frame = QLabel(self.main_widget)
        self.init_ui()

    def init_ui(self):
        """
        the stylesheet of ui
        :return:  none
        """

        # set the size of window
        desktop = QApplication.desktop()
        # get the size of desktop
        desk_width = desktop.availableGeometry().width()
        desk_height = desktop.availableGeometry().height()
        # desk_width = desktop.width()
        # desk_height = desktop.height()
        self.setFixedSize(desk_width, desk_height)

        # add the widgets
        self.frame.setGeometry(QRect(0, 0, desk_width, desk_height))
        self.frame.setPixmap(QPixmap('dog.jpg'))
        self.frame.setScaledContents(True) # filled with content


if __name__ == '__main__':
    app = QApplication([])
    index = Index_UI()
    index.show()
    app.exec_()
