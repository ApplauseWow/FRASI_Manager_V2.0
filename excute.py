# -*-coding:utf-8-*-

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QTreeWidgetItem, QLabel, QWidget, QGraphicsScene
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from GUI import Index, Sys_Option_UI, Identify_Id_UI, Sign_In_UI, Model_Estimate_UI
from sklearn.externals import joblib
import sys
import threading
import time
from Util import *


class Interaction(Index):
    """
    add interaction on the static GUI as the front end
    """

    def __init__(self):
        super(Interaction, self).__init__()
        self.menu.itemClicked[QTreeWidgetItem, int].connect(self.onClicked)
        self.show_hide.clicked.connect(self.hide_show_menu)
        self.id_num.clear()
        self.name.clear()
        self.date.clear()
        self.status.clear()
        self.sys_ui = System()
        self.register_ui = Register()
        self.compare_ui = Compare()
        self.model_estimation = Estimate_Model()
        self.compare_ui.id_exist_signal.connect(self.register_ui.show_cache)
        self.re_signal = True

    def hide_show_menu(self):
        """
        clicked the the screen to show or hide the menu list
        :return: none
        """

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
            camera_thread = threading.Thread(target=Utility.open_camera, args=(self.frame, ))
            camera_thread.daemon = True
            camera_thread.start()
        elif task == u'人脸识别' or task == u"人脸考勤":
            if self.re_signal:
                self.re_signal = False
                rec_thread = threading.Thread(target=Utility.save_cache_of_frame, args=("recognition",))
                rec_thread.daemon = True
                rec_thread.start()
                # use the queue between socket thread and main GUI thread to get the recognition result
                if task == u"人脸考勤":
                    self.status.setText(u"考勤中...")
                    get_thread = threading.Thread(target=self.get_q_data, args=(result_q, True))
                else:
                    self.status.setText(u'识别中...')
                    get_thread = threading.Thread(target=self.get_q_data, args=(result_q, ))
                get_thread.daemon = True
                get_thread.start()
            elif not self.re_signal:
                # back end is recognizing... wait...
                pass
        elif task == u'人脸检索':
            pass
        elif task == u'单脸注册':
            pass
        elif task == u'多脸注册':
            self.compare_ui.exec_() # lock the window until it's closed
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
            self.sys_ui.exec_()
        elif task == u'模型性能可视化':
            self.model_estimation.draw_roc()

    def get_q_data(self, queue, attendance=False):
        """
        to get the data in the queue
        :param queue: which queue
        :param attendance: attendance
        :return: none
        """

        while True:
            if queue.empty():
                pass
            elif not queue.empty():
                try:
                    data = queue.get(timeout=5)
                    if data is not None:
                        self.id_num.setText(data['id'])
                        self.name.setText(data['name'])
                        if not attendance:
                            self.status.setText(u"识别成功")
                        elif attendance:
                            self.date.setText(data['date'])
                            sql = "insert into attendance_record (userid, device_deployment_id, remarks, attendance_status, time) values(%s,%s,%s,%s,%s)"
                            arg_list = [data['id'], 1, '1', '1', data['date']]
                            op = ["insert", sql, arg_list]
                            feedback = Utility.sql_operation(op)
                            if feedback['result'] > 0:
                                self.status.setText(u"考勤成功")
                            else:
                                self.status.setText(u"考勤失败")
                    elif data is None:
                        self.id_num.clear()
                        self.name.clear()
                        self.date.clear()
                        self.status.setText(u"谁在这里")  # 后面有人
                    self.re_signal = True
                    break
                except Queue.Empty as e:
                    print e
                    self.id_num.clear()
                    self.name.clear()
                    self.date.clear()
                    self.status.setText(u"重新操作")
                    self.re_signal = True
                    break

    def keyPressEvent(self, QKeyEvent):
        """
        click the keyboard to do something...
        :param QKeyEvent:
        :return:
        """

        if QKeyEvent.key() == QtCore.Qt.Key_Escape: # clicked the ESC
            Utility.socket_transmission("exit")
            self.setDisabled(True)
            self.close()


class System(Sys_Option_UI):
    """
    override the system option UI
    """

    def __init__(self):
        super(System, self).__init__()
        self.recognition_count.setText(str(RECOGNITION_FRAME))
        self.sign_in_count.setText(str(REGISTER_FRAME))
        self.interval.setText(str(AUTO_SLEEP_INTERIM))
        self.detect_coint.setText(str(DETECT_FRAME))
        self.rejected.connect(self.init_params)
        self.accepted.connect(self.save_params)

    def closeEvent(self, QCloseEvent):
        """
        close the dialog, and initial parameters
        :param QCloseEvent: signal
        :return: none
        """

        self.recognition_count.setText(str(RECOGNITION_FRAME))
        self.sign_in_count.setText(str(REGISTER_FRAME))
        self.interval.setText(str(AUTO_SLEEP_INTERIM))
        self.detect_coint.setText(str(DETECT_FRAME))

    def init_params(self):
        """
        init the params if didn't accept
        :return: none
        """

        self.recognition_count.setText(str(RECOGNITION_FRAME))
        self.sign_in_count.setText(str(REGISTER_FRAME))
        self.interval.setText(str(AUTO_SLEEP_INTERIM))
        self.detect_coint.setText(str(DETECT_FRAME))

    def save_params(self):
        """
        save the params in the dialog
        :return: none
        """

        xml_path = os.path.join(os.getcwd(), "sys.xml")
        params = dict()
        params["recognition_frame"] = self.recognition_count.text()
        params["register_frame"] = self.sign_in_count.text()
        params["auto_sleep_interim"] = self.interval.text()
        params["detect_frame"] = self.detect_coint.text()
        Utility.write_xml(xml_path, params)


class Compare(Identify_Id_UI):
    """
    override the Identify_Id_UI
    to make sure that there exist this identity
    """

    id_exist_signal = QtCore.pyqtSignal(list)

    def __init__(self):
        super(Compare, self).__init__()
        self.id_input.clear()
        self.name_input.clear()
        self.tip.setText(u"确定后，请将脸对准摄像头")
        self.ok.clicked.connect(self.search_for_identity)
        self.cancel.clicked.connect(self.close_window)

    def closeEvent(self, QCloseEvent):
        """
        close the window
        :param QCloseEvent: event signal
        :return: none
        """

        self.id_input.clear()
        self.name_input.clear()

    def close_window(self):
        """
        close the window
        :return: none
        """

        self.close()

    def search_for_identity(self):
        """
        search for the inputted identity from DB
        :return: none
        """

        if self.id_input.text().split() and self.name_input.text().split(): # []
            self.tip.clear()
            sql = "select * from user_inf where userid=%s"
            arg_list = [self.id_input.text()]
            op = ["select", sql, arg_list]
            result = Utility.sql_operation(op)
            if result:
                # exist this identity
                sign_t = threading.Thread(target=Utility.save_cache_of_frame, args=("sign_in", ))
                sign_t.daemon = True
                sign_t.start()
                self.tip.setText(u"请稍后，正在拍照")
                while register_q.empty():  # get the frames for signing in
                    pass
                args = register_q.get(timeout=5)
                if args is None:
                    self.tip.setText(u"请重新提交，并对准摄像头")
                elif args is not None:
                    data_list = [args, self.id_input.text(), self.name_input.text()]
                    self.id_exist_signal.emit(data_list)
                    self.id_input.clear()
                    self.accept()
            else:
                self.tip.setText(u"信息不存在，请到后台注册授权")
        else:
            self.tip.setText(u"不能为空！")


class Register(Sign_In_UI):
    """
    override the Sign_In_UI
    ask custom to select the frames to register
    """

    training_signal = QtCore.pyqtSignal(list)

    def __init__(self):
        super(Register, self).__init__()
        self.ok.clicked.connect(self.training)
        self.data_list = list()

    def show_cache(self, arg_list):
        """
        show the caches
        :param arg_list: some arguments (fave_list, id, name)
        :return: none
        """

        self.data_list = arg_list
        face_list = arg_list[0]
        list_len = len(face_list)

        positions = [(0, y) for y in range(list_len)]
        self.filewidget = QWidget()
        self.filewidget.setMinimumSize(350, 1800)
        for position, name in zip(positions, face_list):
            lab = QLabel(self.filewidget)
            lab.setFixedSize(350, 200)
            _frame = imread(name)
            # cvtColor(_frame, COLOR_BGR2RGB, _frame)
            row = _frame.shape[0]
            col = _frame.shape[1]
            bytesPerLine = _frame.shape[2] * col
            lab.setPixmap(QPixmap.fromImage(QImage(_frame.data, col, row, bytesPerLine, QImage.Format_RGB888)).scaled(lab.width(), lab.height()))
            lab.move(300 * position[0] + 50, 250 * position[1] + 70)
        self.scrollArea.setWidget(self.filewidget)
        self.exec_()

    def training(self):
        """

        :param face_list:
        :return:
        """

        self.training_signal.connect(self.send_list)
        self.training_signal.emit(self.data_list)

    def send_list(self, data):
        """

        :param data:[0] -> face [1] -> id
        :return: none
        """

        print "will send the list..."
        send_data = dict()
        send_data['register_data'] = data
        threading.Thread(target=Utility.socket_transmission, args=(pickle.dumps(send_data),)).start()
        self.accept()


class Estimate_Model(Model_Estimate_UI):
    """
    override the Model_Estimate_UI
    show the ROC of the model
    """

    def __init__(self):
        super(Estimate_Model, self).__init__()
        self.pca_model = joblib.load(os.path.join(os.getcwd(), "Training_data", "pca"))
        self.clf = joblib.load(os.path.join(os.getcwd(), "Training_data", "classifier_svm"))
        self.persons = Utility.read_param_from_xml(os.path.join(os.getcwd(), "Training_data","person.xml"))

        def person_id(n):
            n = bytes.decode(n)
            return self.persons[n]
        self.data = np.loadtxt(os.path.join(os.getcwd(), "Training_data", "test_data.dat"), delimiter=",", converters={128: person_id}, dtype=float)
        x, y = np.split(self.data, (128,), axis=1)

        self.X = self.pca_model.transform(x)
        self.Y = y
        self.pred_y = self.clf.predict(self.X)
        self.scores = self.clf.decision_function(self.X)
        class_ = [x for k, x in self.persons.items()]
        if len(class_) > 2:
            self.Y = label_binarize(self.Y, classes=class_)
        else:  # 二分类不用二值化
            self.Y = np.array(map(lambda x: 0 if x == -1 else 1, class_))

    def draw_roc(self):
        """
        draw the roc of the model
        :return:
        """

        fpr, tpr, threshold = roc_curve(self.Y.ravel(), self.scores.ravel())
        roc_auc = auc(fpr, tpr)
        import matplotlib
        matplotlib.use("Qt5Agg")  # 声明使用QT5
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        fig = Figure(figsize = (11, 5), dpi=100)
        fig_canvas = FigureCanvas(fig)
        fig.add_subplot(111).plot(fpr, tpr, color='darkorange',
                     lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        graphic_scene = QGraphicsScene()
        graphic_scene.addWidget(fig_canvas)  # 创建一个QGraphicsScene，因为加载的图形（FigureCanvas）不能直接放到graphicview控件中，必须先放到graphicScene，然后再把graphicscene放到graphicview中
        self.canvas.setScene(graphic_scene)
        self.canvas.show()
        self.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # create the instance of GUI to show it
    front_end = Interaction()
    front_end.show()
    camera_thread = threading.Thread(target=Utility.open_camera, args=(front_end.frame, ))
    # static_ui = Index()
    # static_ui.show()
    # camera_thread = threading.Thread(target=Utility.open_camera, args=(static_ui.frame, ), name="camera")
    camera_thread.daemon = True
    camera_thread.start()
    sys.exit(app.exec_())