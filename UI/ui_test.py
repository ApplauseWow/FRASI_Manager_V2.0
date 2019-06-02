# coding=utf-8
# -*-coding:utf-8-*-
from FRASI_Manager_V2.UI.index import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow, QTreeWidgetItem
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from cv2.cv2 import imread, cvtColor, COLOR_BGR2RGB


class Index(QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(Index, self).__init__()
        self.setupUi(self)

        # the group of widgets >>>
        self.info_list = (self.nameLabel, self.nameLineEdit,
                          self.iDLabel, self.iDLineEdit,
                          self.dateLabel, self.dateLineEdit,
                          self.status)

        # structure of ui >>>
        self.setWindowFlags(Qt.FramelessWindowHint)
        # self.setAttribute(Qt.WA_TranslucentBackground) # transparent
        desktop = QApplication.desktop()
        desk_width = desktop.availableGeometry().width()
        desk_height = desktop.availableGeometry().height()
        # desk_width = desktop.width()
        # desk_height = desktop.height()
        self.resize(desk_width, desk_height)
        self.menu.expandAll()
        self.menu.setFixedWidth(desk_width // 8)
        # self.menu_list.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        # self.menu_list.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.frame.resize(desk_width - self.menu.width(), desk_height)
        self.show_hide.setFixedSize(self.menu.width() // 10, desk_height)
        self.show_hide.setText(">")
        # <<< structure of ui

        # debugging of ui >>>
        # s = self.menu_list.size() # for debugging
        # r = self.frame.size() # for debugging
        '''
        the image will not be found by the string path 
        unless use qrc file to show the path 
        or use the openCV (read file) to add the image into the memory
        '''
        self.status.setText(u'正在考勤')
        img = imread('dog.jpg') # for debugging
        cvtColor(img, COLOR_BGR2RGB, img)
        row = img.shape[0] # for debugging
        col = img.shape[1] # for debugging
        bytesPerLine = img.shape[2] * col # for debugging
        pix = QPixmap.fromImage(QImage(img.data, col, row, bytesPerLine, QImage.Format_RGB888).scaled(self.frame.width(), self.frame.height())) # for debugging
        self.frame.setPixmap(pix) # for debugging
        self.frame.setStyleSheet("border: 2px solid red") # for debugging
        self.menu.setStyleSheet("border: 2px solid red")  # for debugging
        self.show_hide.setStyleSheet("border: 2px solid red")  # for debugging
        # <<< debugging of ui

        # signal and slot of widgets >>>
        self.show_hide.clicked.connect(self.hide_show_menu)
        self.menu.itemClicked[QTreeWidgetItem, int].connect(self.onClicked)
        # <<< signal and slot of widgets

        self.frame.setScaledContents(True)
        self.showFullScreen()
        '''
        QLabel.setText will check if it's pure text or html
        so the parameter -> str can be the html tag and the css stylesheet
        '''

    def hide_show_menu(self):
        """
        clicked the the screen to show or hide the menu list
        :return: none
        """

        # if self.menu.sizePolicy().horizontalPolicy() == QSizePolicy.Preferred:
        #     self.menu.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        #     self.show_hide.setText("<")
        # elif self.menu.sizePolicy().horizontalPolicy() == QSizePolicy.Ignored:
        #     self.menu.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        #     self.show_hide.setText(">")
        if self.menu.isVisible():
            # is visible and hide it
            self.menu.hide()
            map(lambda obj: obj.hide(), self.info_list)
            self.show_hide.setText("<")
        elif not self.menu.isVisible():
            # is invisible and show it
            self.menu.show()
            map(lambda obj: obj.show(), self.info_list)
            self.show_hide.setText(">")

    def onClicked(self, item, column):
        """
        when click the item in menu, trigger this function
        :param item: object of item
        :param column: number of colum
        :return: none
        """

        task = item.text(0)
        print task
        # print type(task) # object of task is unicode
        if task == u'人脸检测':
            pass
        elif task == u'人脸识别' or task == u"人脸考勤":
            pass
        elif task == u'人脸检索':
            pass
        elif task == u'单脸注册':
            pass
        elif task == u'多脸注册':
            pass
        elif task == u'身份证注册':
            pass
        elif task == u'语音识别':
            pass
        elif task == u'语音检索':
            pass
        elif task == u'语音考勤':
            pass
        elif task == u'语音注册':
            pass
        elif task == u'数据导出':
            pass
        elif task == u'数据导入':
            pass
        elif task == u'统计查看':
            pass
        elif task == u'参数配置':
            pass

    def keyPressEvent(self, QKeyEvent):
        """
        click the keyboard to do something...
        :param QKeyEvent:
        :return:
        """

        if QKeyEvent.key() == Qt.Key_Escape: # clicked the ESC
            self.setDisabled(True)
            self.close()


if __name__ == '__main__':
    app = QApplication([])
    index = Index()
    index.show()
    app.exec_()