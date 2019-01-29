from index import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow, QSizePolicy
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from cv2.cv2 import imread


class Index(QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(Index, self).__init__()
        self.setupUi(self)
        self.setWindowFlags(Qt.FramelessWindowHint)
        desktop = QApplication.desktop()
        desk_width = desktop.availableGeometry().width()
        desk_height = desktop.availableGeometry().height()
        # desk_width = desktop.width()
        # desk_height = desktop.height()
        self.resize(desk_width, desk_height)
        self.menu_list.expandAll()
        self.menu_list.setFixedWidth(desk_width // 8)
        self.menu_list.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.frame.resize(desk_width - self.menu_list.width(), desk_height)
        # self.menu_list.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        # s = self.menu_list.size() # for debugging
        # r = self.frame.size() # for debugging
        '''
        the image will not be found by the string path 
        unless use qrc file to show the path 
        or use the openCV (read file) to add the image into the memory
        '''
        img = imread('dog.jpg') # for debugging
        row = img.shape[0] # for debugging
        col = img.shape[1] # for debugging
        bytesPerLine = img.shape[2] * col # for debugging
        pix = QPixmap.fromImage(QImage(img.data, col, row, bytesPerLine, QImage.Format_RGB888).scaled(self.frame.width(), self.frame.height())) # for debugging
        self.frame.setPixmap(pix) # for debugging
        self.frame.setStyleSheet("border: 2px solid red") # for debugging
        self.frame.setScaledContents(True)
        self.showFullScreen()
        '''
        QLabel.setText will check if it's pure text or html
        so the parameter -> str can be the html tag and the css stylesheet
        '''


if __name__ == '__main__':
    app = QApplication([])
    index = Index()
    index.show()
    app.exec_()