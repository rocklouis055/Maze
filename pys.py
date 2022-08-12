import sys
import time
from PyQt5.uic import loadUi 
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog,QApplication,QWidget,QStackedWidget

class WelcomeScreen(QDialog):
    def __init__(self):
        super(WelcomeScreen,self).__init__()
        loadUi("louise.ui",self)
app=QApplication(sys.argv)
welcome=WelcomeScreen()
widget=QStackedWidget()
widget.addWidget(welcome)
widget.setFixedHeight(602)
widget.setFixedWidth(810)
widget.show()
time.sleep(5)