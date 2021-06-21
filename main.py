import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import uic
import numpy as np
import cv2
import dlib
from threading import Thread
from functools import reduce
from toolkit.mask import put_on_sunglass, put_on_neko_ears
from toolkit.filter import cartoonize, sketchify

form_class = uic.loadUiType("ui.ui")[0]
faceDetector = dlib.get_frontal_face_detector()


class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.check_neko_ears:  QCheckBox
        self.check_sunglass:   QCheckBox

        self.check_cartoon:    QCheckBox
        self.check_sketch:     QCheckBox
        self.check_invert:     QCheckBox

        self.pushButton:       QPushButton

        self.pushButton.clicked.connect(self.showOpenFileDialog)

    def showOpenFileDialog(self):
        fname = QFileDialog.getSaveFileName(self, 'Save file', './')
        filepath = fname[0]
        print(filepath)


video_capture = cv2.VideoCapture(0)


def show():
    while True:
        _, frame = video_capture.read()
        frame = np.array(np.fliplr(frame))
        h, w, c = frame.shape

        faces = faceDetector(frame)

        if main_form.check_neko_ears.isChecked():
            frame = reduce(lambda acc, face: put_on_neko_ears(
                acc, face), faces, frame)

        if main_form.check_sunglass.isChecked():
            frame = reduce(lambda acc, face: put_on_sunglass(
                acc, face), faces, frame)

        if main_form.check_invert.isChecked():
            frame = ~frame

        if main_form.check_sketch.isChecked():
            frame = sketchify(frame)

        if main_form.check_cartoon.isChecked():
            frame = cartoonize(frame)

        qImg = QImage(np.array(frame[:, :, ::-1]),
                      w, h, w*c, QImage.Format_RGB888)
        main_form.label.setPixmap(QPixmap.fromImage(qImg))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_form = WindowClass()
    main_form.show()
    t1 = Thread(target=show)
    t1.daemon = True
    t1.start()
    app.exec_()
