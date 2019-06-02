# -*-coding:utf-8-*-
from sklearn.externals import joblib
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from face_recognition import face_locations, face_encodings
from PyQt5.QtGui import QPixmap, QImage
import socket
import xml.etree.ElementTree as EL
import numpy as np
from cv2.cv2 import *
import threading
import time
import dlib
import os
import copy
import pymysql
from functools import wraps
import Queue
try:
    import cPickle as pickle
except:
    import pickle

# preparation
# -parse parameters
path = "./sys.xml"
tree = EL.parse(path)
root = tree.getroot()
param_dict = dict()

for param in root.iter("param"):
    name = param.attrib['name']
    count = param.attrib['count']
    param_dict[name] = count

# -clear the cache
file_list = map(lambda x: map(lambda y: os.path.join(x, y), os.listdir(x)), map(lambda x:os.path.join(os.getcwd(), "Cache", x), ["detect", "recognition", "sign_in"]))
map(lambda x: map(lambda y: os.remove(y), x), file_list)

# parameters of system
# to avoid restarting the system after modify the system parameters,
# can modify parameters in both of the memory and xml file
RECOGNITION_FRAME = param_dict["recognition_frame"] # the number of captured frames for recognition
REGISTER_FRAME = param_dict["register_frame"] # the number of captured frames for register
AUTO_SLEEP_INTERIM = param_dict["auto_sleep_interim"] # the seconds of interim
DETECT_FRAME = param_dict["detect_frame"] # the number of captured frames for detect

SWITCH = True # the status of camera - T => open F => close
DET_CACHE_SIGNAL = False # permission for saving detect cache
REC_CACHE_SIGNAL = False # permission for saving recognition cache
REG_CACHE_SIGNAL = False # permission for saving register cache

# the queue between the socket and GUI thread for getting the results
result_q = Queue.Queue(1)
# the queue between the socket and GUI thread for selecting frames
register_q = Queue.Queue(1)

# following functions in utilities class are scalable and pluggable
# process runs in backend


class Utility(object):
    """
    utilities
    """

    @staticmethod
    def open_camera(gui_frame):
        """
        - front end
        open the camera and capture each frame to show on the GUI label
        :param gui_frame: the object of GUI label
        :return: none
        """

        # search the available device
        for i in range(4):
            device = VideoCapture(i)
            if device.isOpened():
                # device is available
                print("device connect successfully!")
                global SWITCH
                SWITCH = True
                # start timer
                # threading.Thread(target=Utility.camera_timer, args=(device, float(AUTO_SLEEP_INTERIM))).start()
                threading.Thread(target=Utility.socket_transmission, args=("timer", )).start()

                NUM_RECOGNITION_CACHE = 0  # the number of saved frames for recognition
                NUM_DETECT_CACHE = 0  # the number of saved frames for detect
                NUM_SIGN_IN_CACHE = 0  # the number of saved frames for signing in

                while True:
                    if not SWITCH:
                        # close the camera
                        device.release()
                        return
                    # capture frame
                    success, _frame = device.read()
                    if not success:  # not captured  successfully
                        msg = "no frame was captured!"
                        print("no frame was captured!\nstop working...")
                        return
                    elif success:  # captured successfully
                        # data of frame
                        row = _frame.shape[0]
                        col = _frame.shape[1]
                        bytesPerLine = _frame.shape[2] * col

                        # image processing
                        # frame_flipped = flip(frame, 1)  # 0-vertical 1-horizontal -1-ver&hor
                        for i in range(col):
                            _frame[:, [i, col - 1 - i]] = _frame[:, [col - 1 - i, i]]
                            if (col - 1 - i - i) <= 2:
                                break
                        # thread of recognition
                        cvtColor(_frame, COLOR_BGR2RGB, _frame)

                        # store cache of frames
                        global DET_CACHE_SIGNAL
                        global REC_CACHE_SIGNAL
                        global REG_CACHE_SIGNAL
                        if DET_CACHE_SIGNAL:
                            print "save the cache of the frame for detect..."
                            img_path = os.path.join("./Cache/detect", "det_cache_" + str(NUM_DETECT_CACHE) + ".jpg")
                            # img_path = ("./Cache/detect/det_cache_".join(NUM_DETECT_CACHE)).join(".jpg")
                            imwrite(img_path, _frame)
                            NUM_DETECT_CACHE += 1
                            if NUM_DETECT_CACHE == int(DETECT_FRAME):
                                # have finished saving cache and init the var
                                DET_CACHE_SIGNAL = False
                                NUM_DETECT_CACHE = 0
                                # send msg -- have saved cache and start detect
                                threading.Thread(target=Utility.socket_transmission, args=("detect", )).start()
                        if REC_CACHE_SIGNAL :
                            print "save the cache of the frame for recognition..."
                            img_path = os.path.join("./Cache/recognition", "rec_cache_" + str(NUM_RECOGNITION_CACHE) + ".jpg")
                            # img_path = ("./Cache/recognition/rec_cache_".join(NUM_RECOGNITION_CACHE)).join(".jpg")
                            imwrite(img_path, _frame)
                            NUM_RECOGNITION_CACHE += 1
                            if NUM_RECOGNITION_CACHE == int(RECOGNITION_FRAME):
                                REC_CACHE_SIGNAL = False
                                NUM_RECOGNITION_CACHE = 0
                                # send msg -- have saved cache and start recognizing
                                threading.Thread(target=Utility.socket_transmission, args=("recognition", )).start()
                        if REG_CACHE_SIGNAL :
                            print "save the cache of the frame for signing in..."
                            img_path = os.path.join("./Cache/sign_in", "sign_cache_" + str(NUM_SIGN_IN_CACHE) + ".jpg")
                            # img_path = ("./Cache/sign_in/sign_cache_".join(NUM_SIGN_IN_CACHE)).join(".jpg")
                            copy_frame = copy.deepcopy(_frame)
                            copy_frame = Utility.rotation(copy_frame, (1 if NUM_SIGN_IN_CACHE % 2 == 0 else -1)*NUM_SIGN_IN_CACHE*3)
                            imwrite(img_path, copy_frame)
                            NUM_SIGN_IN_CACHE += 1
                            if NUM_SIGN_IN_CACHE == int(REGISTER_FRAME):
                                REG_CACHE_SIGNAL = False
                                NUM_SIGN_IN_CACHE = 0
                                threading.Thread(target=Utility.socket_transmission, args=("sign_in", )).start()
                        # show on the GUI
                        gui_frame.setPixmap(QPixmap.fromImage(QImage(_frame.data,
                                                                     col, row, bytesPerLine,
                                                                     QImage.Format_RGB888)).scaled(gui_frame.width(),
                                                                                                   gui_frame.height()))
            elif not device.isOpened():
                # device is not available
                msg = "device is not available!"
                print("device is not available!")
                device.release()
        else:
            # none of the devices is available
            msg = "none of the devices is available"
            print("none of the devices is available")

    @staticmethod
    def camera_timer(seconds, conn):
        """
        - backend
        close the camera after some interim
        :param seconds: interim
        :param conn: backend socket for sending message
        :return: none
        """

        print "start counting seconds..."
        time.sleep(seconds)
        # if there is a detected face pass
        # or not so, release the object of camera
        # send msg -- start storing the cache
        cmd_dict = {"save_cache": "detect"}
        data= pickle.dumps(cmd_dict)
        conn.sendall(data)

    @staticmethod
    def save_cache_of_frame(kind):
        """
        - front end
        save the all kinds of frames in Cache directory for specific use
        :param kind: according to the task name, save the frames
        :return: none
        """

        if kind == "detect":
            # save the cache for detect
            global DET_CACHE_SIGNAL
            DET_CACHE_SIGNAL = True
        elif kind == "recognition":
            # save the cache for recognition
            global REC_CACHE_SIGNAL
            REC_CACHE_SIGNAL = True
        elif kind == "sign_in":
            # save the cache for register
            global REG_CACHE_SIGNAL
            REG_CACHE_SIGNAL = True

    @staticmethod
    def detect_face(img_path, conn):
        """
        - backend
        detect the cache of frames if there is face
        :param img_path: detect the image in the directory of the path
        :param conn: the object of socket
        :return: boolean => true : exist false : none
        """

        detector = dlib.get_frontal_face_detector()
        for _file in os.listdir(img_path):
            if _file == "":
                # dir is empty
                # conn.sendall(bytes("no_file"))
                conn.sendall(pickle.dumps("no_file"))
            else:
                file_path = os.path.join(img_path, _file)
                img = imread(file_path)
                dets = detector(img, 1)
                if len(dets) > 0:
                    # there is detected face
                    print "exist face!"
                    # conn.sendall(bytes("exist"))
                    conn.sendall(pickle.dumps("exist"))
                    break
                elif len(dets) == 0:
                    print "no face here!"
        else:
            # no face in each frame
            # conn.sendall(bytes("no_face"))
            conn.sendall(pickle.dumps("no_face"))

    @staticmethod
    def detect_a_face(frame):
        """
        - backend
        just for detect a frame
        :param frame: a frame
        :return: true or false
        """

        detector = dlib.get_frontal_face_detector()
        det = detector(frame, 1)
        if len(det) == 1:
            return True
        else:
            return False

    @staticmethod
    def recognition(img_path, conn):
        """
        - backend
        face recognition
        :param img_path: a list containing the path of processed frames (have detected => exist face)
        :param conn: socket
        :return: none
        """

        clf_path = os.path.join(os.getcwd(), "Training_data", "classifier_svm")
        pca_path = os.path.join(os.getcwd(), "Training_data", "pca")
        clf = joblib.load(clf_path)
        pca_model = joblib.load(pca_path)
        rec_item = list()
        result_dict = dict()

        for _file in os.listdir(img_path):
            file_path = os.path.join(img_path, _file)
            # start = time.time() hog:each consumes 0.66s-0.68s and without GPU cnn:each consumes 0.55-0.6s gpu:56%
            img = imread(file_path)
            if Utility.detect_a_face(img):
                # there is a face
                # face_location = face_locations(img)
                _face_location = list()
                face_location = face_locations(img, model="cnn")
                _face_location.append(face_location[0]) # code here is time-consuming !
                face_encoding = face_encodings(img, _face_location) # or here !
                # print type(face_encoding) <type 'list'>
                encoding = np.asarray(face_encoding, np.float32)
                encoding = pca_model.transform(encoding)
                # start = time.time() # cnn:each consumes 0.00018-0.00032s hog: similar
                _id = int(clf.predict(encoding)) # sometimes detect 1 face here but locate 2 face feature;so must choose [0] and append to a list
                # end = time.time()
                # print "show time:-------"
                # print end-start
                rec_item.append(_id)
            else:
                # none face here or many faces here
                pass

        # finish recognizing and count
        for res in rec_item:
            result_dict[res] = result_dict.get(res, 0) + 1 # dict.get(key, initial_num)
        sorted_list = sorted(result_dict.items(), key= lambda x:x[1], reverse=True)
        for key, value in sorted_list:
            # print key, type(key)
            # send the recognition result
            if key == -1: # unknown
                result = {"rec_result": None}
                conn.sendall(pickle.dumps(result))
            else:
                info_dict = dict()
                sql = "select name from user_inf where userid=%s"
                op = ["select", sql, [str(key)]]
                info = Utility.sql_operation(op)
                info['id'] = str(key)
                info['date'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                result = {"rec_result": info}
                conn.sendall(pickle.dumps(result))
            break
        else:
            result = {"rec_result": None}
            conn.sendall(pickle.dumps(result))

    @staticmethod
    def sign_in(img_path, conn):
        """
        - backend
        sign in
        :param img_path: path of cache
        :param conn: socket
        :return: none
        """

        face_list = list()
        for _file in os.listdir(img_path):
            file_path = os.path.join(img_path, _file)
            frame = imread(file_path)
            if Utility.detect_a_face(frame):
                face_list.append(file_path)
            else:
                pass
        data = dict()
        if len(face_list) == 0: # no suitable frame
            data["show_cache"] = None
        else:
            data["show_cache"] = face_list
        conn.sendall(pickle.dumps(data))

    @staticmethod
    def training(face_list, _id, name, conn):
        """
        training the model
        :param face_list: the path of frame
        :param _id: identity
        :param conn: socket
        :param name: name
        :return: none
        """

        feature_path = os.path.join(os.getcwd(), "Training_data", "data.dat")
        test_path = os.path.join(os.getcwd(), "Training_data", "test_data.dat")
        cls_path = os.path.join(os.getcwd(), "Training_data")
        xml_path = os.path.join(os.getcwd(), "Training_data","person.xml")
        persons = Utility.read_param_from_xml(xml_path)
        if name in persons or _id in persons.values():
            # have existed, won't train
            pass
        else:
            person = dict()
            person["name"] = name
            person["_id"] = _id
            persons[name] = _id
            Utility.write_xml(xml_path, person)

            def person_id(n):
                n = bytes.decode(n)
                return persons[n]

            for _file, i in zip(face_list, [x for x in range(len(face_list))]):
                # ret, frame = cv2.VideoCapture(0).read()
                face = imread(_file)
                face = resize(face, (0, 0), fx=0.25, fy=0.25)
                location = face_locations(face, model="cnn")
                encoding = face_encodings(face, location)
                d_path = os.path.join(os.getcwd(), "Training_data", i.__str__() + '.txt')
                np.savetxt(d_path, encoding, delimiter=",")
                with open(d_path, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip("\n") + "," + name + "\n"
                        with open((feature_path if i != (len(face_list) - 1) else test_path),"a") as e:  # the last one for test
                            e.write(line)

            data = np.loadtxt(feature_path, delimiter=",", converters={128: person_id}, dtype=float)
            x, y = np.split(data, (128,), axis=1)
            # x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=67, train_size=1)

            pca_model = PCA(n_components=0.8, whiten=True, random_state=67).fit(x)
            pca_x = pca_model.transform(x)

            classifier = svm.SVC(C=10, gamma=20, kernel='rbf', decision_function_shape='ovr', probability=True, random_state=67)
            classifier.fit(pca_x, y)
            # s = classifier.score(pca_x, y)
            # p = classifier.predict(pca_x)
            # pp = classifier.predict_proba(pca_x)
            joblib.dump(classifier, os.path.join(cls_path, "classifier_svm"))
            joblib.dump(pca_model, os.path.join(cls_path, "pca"))

        conn.sendall(pickle.dumps("train_done"))

    @staticmethod
    def socket_transmission(task):
        """
        - front end
        connected with backend via socket and transmit the information of task
        this function can be normalized for specific task like JAVA servlet
        :param task: a string containing the name of task
        :return: none
        """

        obj = socket.socket()
        obj.connect(("127.0.0.1", 44967))

        ret_bytes = obj.recv(1024)
        ret_str = str(ret_bytes)
        if ret_str == "got":
            print "backend is working, keep accepting task..."
            obj.sendall(bytes(task))
            ret_bytes = pickle.loads(obj.recv(1024))
            # print type(ret_bytes)
            if type(ret_bytes) == type(str()):
                ret_str = str(ret_bytes)
                # feedback
                if ret_str == "no_file":
                    # no cache exist
                    print "no cache file"
                    obj.close()
                elif ret_str == "exist":
                    # cant't close the camera and start timing again
                    threading.Thread(target=Utility.socket_transmission, args=("timer",)).start()
                    obj.close()
                elif ret_str == "no_face":
                    # close the camera
                    global SWITCH
                    SWITCH = False
                    obj.close()
                elif ret_str == "train_done":
                    print "have finished training..."
                elif ret_str == "exiting":
                    print "backend is stopping..."
            elif type(ret_bytes) == type(dict()):
                # feedback
                for key, value in ret_bytes.items():
                    if key == "save_cache":
                        Utility.save_cache_of_frame(value)
                    elif key == "rec_result":
                        print value
                        # value is a information dictionary
                        if not result_q.empty():
                            result_q.get()  # clear the queue
                        result_q.put(value)
                    elif key == "show_cache":
                        if not register_q.empty():
                            register_q.get()
                        register_q.put(value)
                    else:
                        pass
        else:
            print "there is something wrong with backend...\nfail to connect"
        obj.close()

    @staticmethod
    def sql_operation(op):
        """
        - backend
        SQL operation
        :param op: a list : op[0] => operation op[1] => sql, op[2]: args list
        :return: result dictionary
        """

        _host = "localhost"
        _user = "root"
        _pwd = "123qwe"
        _db = "frasi"
        arg_list = op[2]
        sql = op[1]
        operation = op[0]
        result_dict = dict()

        try:
            db = pymysql.connect(host=_host, user=_user, password=_pwd, db=_db, port=3306, charset="utf8")
            with db:
                # sql operation
                print "db connected..."
                cursor = db.cursor()
                if operation == "select":
                    cursor.execute(sql, args=arg_list)
                    print "select..."
                    result = cursor.fetchall()
                    for row in result:
                        _name = row[0]
                        result_dict['name'] = _name
                elif operation == "insert":
                    effect_row = cursor.execute(sql, args=arg_list)
                    print type(effect_row)
                    if effect_row > 0:
                        result_dict['result'] = effect_row
                    else:
                        result_dict['result'] = 0
                    db.commit()
                cursor.close()
        except Exception as e:
            print e
        return result_dict

    @staticmethod
    def read_param_from_xml(xml_path):
        """
        parse the XML file for getting the parameters of system
        :parameter xml_path: path of xml file
        :return: params_dict that a dictionary containing the parameters of system
        """

        tree = EL.parse(xml_path)
        root = tree.getroot()
        params_dict = dict()

        if "person.xml" in xml_path.split("/"): # training label reflection
            for p in root.findall('info'):
                _name = p.get('name')
                _id = p.get('_id')
                params_dict[_name] = int(_id)
        elif "sys.xml" in xml_path.split("/"): # system parameters
            for param in root.iter("param"):
                _name = param.attrib['name']
                _count = param.attrib['count']
                params_dict[_name] = _count

        return params_dict

    @staticmethod
    def write_xml(xml_path, param_dict):
        """
        rewrite the xml file
        :param xml_path: path of xml file
        :return: none
        """

        tree = EL.parse(xml_path)
        root = tree.getroot()

        if "sys.xml" in xml_path.split("/"):
            for param in root:
                _name = param.attrib["name"]
                print _name
                # the name must be same!
                param.set("count", param_dict[_name])
        elif "person.xml" in xml_path.split("/"):
            child = EL.Element('info', {'name': param_dict["name"], '_id': param_dict["_id"]})
            root.append(child)
        tree.write(xml_path)

    @staticmethod
    def rotation(image, angle, center=None, scale=1.0):
        """
        - inner function
        for rotation image
        :param angle:
        :param center:
        :param scale:
        :return:the rotated image
        """
        (h, w) = image.shape[:2]
        if center is None:
            center = (w // 2, h // 2)
        M = getRotationMatrix2D(center, angle, scale)
        rotated = warpAffine(image, M, (w, h))
        return rotated

    def fn_timer(function):
        """
        - test
        for test the run time
        :param function: function
        :return:
        """
        @wraps(function)
        def function_timer(*args, **kwargs):
            t0 = time.time()
            result = function(*args, **kwargs)
            t1 = time.time()
            print ("Total time running %s: %s seconds" %
                   (function.func_name, str(t1 - t0))
                   )
            return result

        return function_timer

