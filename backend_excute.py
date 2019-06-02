# -*-coding:utf-8-*-

import SocketServer
from Util import *


class Backend(SocketServer.BaseRequestHandler):

    def __init__(self, request, client_address, server):
        self.request = request
        self.client_address = client_address
        self.server = server
        self.setup()
        try:
            self.handle()
        finally:
            self.finish()

    def handle(self):

        conn = self.request
        conn.sendall(bytes("got"))
        while True:
            ret_bytes = conn.recv(1024) # receive the bytes stream
            ret_str = str(ret_bytes) # transport the type

            #  match the task
            if ret_str == "recognition":
                print "got it, recognition..."
                img_path = os.path.join(os.getcwd(), "Cache", "recognition")
                threading.Thread(target=Utility.recognition, args=(img_path, conn)).start()
            elif ret_str == "sign_in":
                print "got it, sign in..."
                img_path = os.path.join(os.getcwd(), "Cache", "sign_in")
                threading.Thread(target=Utility.sign_in, args=(img_path, conn)).start()
            elif ret_str == "timer":
                print "got it, timing..."
                threading.Thread(target=Utility.camera_timer, args=(float(AUTO_SLEEP_INTERIM),conn)).start()
            elif ret_str == "detect":
                print "got it, detect..."
                img_path = os.path.join(os.getcwd(), "Cache", "detect")
                threading.Thread(target=Utility.detect_face, args=(img_path, conn)).start()
            elif ret_str == "exit":
                print 'exit'
                conn.sendall(pickle.dumps("exiting"))
                self.server.shutdown()
                self.request.close()
                break
            else:
                # receive data
                # type(pickle.dumps(dict() or list())) = 'str'
                try:
                    data = pickle.loads(ret_str)
                    print "got it face list..."
                    for key, value in data.items():
                        if key == "register_data":
                            face_list = value[0]
                            _id = value[1]
                            name = value[2]
                            threading.Thread(target=Utility.training, args=(face_list, _id, name, conn)).start()
                        else:
                            pass
                except EOFError as e:
                    pass


if __name__ == "__main__":
    server = SocketServer.ThreadingTCPServer(("127.0.0.1", 44967), Backend)
    server.serve_forever()

