# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 20:14:19 2019

@author: BurakBey
"""



import cv2
 
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys
import scipy
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import os
import glob

import timeit


class ExampleContent(QWidget):
    def __init__(self, parent,fileName1,fileName2=''):
        self.parent = parent
        
        self.labInput= QLabel()
        self.labResult= QLabel()
        
        self.qpInput = None
        self.qpResult = None
        
        QWidget.__init__(self, parent)
        self.initUI(fileName1,fileName2)
        
        
    def initUI(self,fileName1,fileName2=''):        

        groupBox1 = QGroupBox('Input Image')
        self.vBox1 = QVBoxLayout()        
        groupBox1.setLayout(self.vBox1)
        
        groupBox2 = QGroupBox('Result Image')
        self.vBox2 = QVBoxLayout()
        groupBox2.setLayout(self.vBox2)        
        
        hBox = QHBoxLayout()
        hBox.addWidget(groupBox1)
        hBox.addWidget(groupBox2)

        self.setLayout(hBox)
        self.setGeometry(0, 0, 0,0)
        
        self.InputImage(fileName1)
        if fileName2 != '':
            self.ResultImage(fileName2)    
                
    def InputImage(self,fN):
        
        self.qpInput = QPixmap(fN)
        self.labInput.setPixmap(self.qpInput)
        self.vBox1.addWidget(self.labInput)

    def ResultImage(self,fN):
        self.qpResult = QPixmap(fN)
        self.labResult.setPixmap(self.qpResult)
        self.vBox2.addWidget(self.labResult)
        
class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.title = "CornerDetector-Segmentor"
        self.top = 1000
        self.left = 200
        self.width = 500
        self.height = 500

        self.eigenfaces_DB = ''
        self.optic_DB = ''
        
        self.initWindow()
        
    def initWindow(self):
         
        exitAct = QAction(QIcon('exit.png'), '&Exit' , self)
        importAct = QAction('&Open Input' , self)        
        opticalflowAction = QAction('&Optical Flow' , self)
        eigenfacesAction = QAction('&EigenFaces' , self)
        
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        importAct.setStatusTip('Open Input')
                
        exitAct.triggered.connect(self.closeApp)
        importAct.triggered.connect(self.importInput)
        opticalflowAction.triggered.connect(self.opticalFlowAction)
        eigenfacesAction.triggered.connect(self.eigenFacesAction)
        
        self.statusBar()
        
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAct)
        fileMenu.addAction(importAct)
        
        self.content = ExampleContent(self, '', '')
        self.setCentralWidget(self.content)
        
        self.cornerToolbar = self.addToolBar('Optical Flow')
        self.cornerToolbar2 = self.addToolBar('EigenFaces')
        self.cornerToolbar.addAction(opticalflowAction)
        self.cornerToolbar2.addAction(eigenfacesAction)
        
        self.setWindowTitle(self.title)
        self.setStyleSheet('QMainWindow{background-color: darkgray;border: 1px solid black;}')
        self.setGeometry( self.top, self.left, self.width, self.height)
        self.show()
    
    def closeApp(self):
        sys.exit()
        
    def importInput(self,db_type = 0):
        fileName = QFileDialog.getExistingDirectory(self, "Select a Database Path")
        if( fileName == ''):
            return
        
        if(db_type == 0):
            self.optic_DB = fileName
        elif(db_type==1):
            self.eigenfaces_DB = fileName
        else:
            print('Error encountered while selecting the DB')
            
    def opticalFlowAction(self):
        self.importInput(0)
        print(self.optic_DB)
    def eigenFacesAction(self):
        self.importInput(1)
        X = self.readDatabase()
        avgIm = self.findAverageImage(X)
        print(avgIm)
        cv2.imwrite('../a.png' , avgIm)
        Z = self.createDifferenceImages(avgIm,X)
        print(Z.shape)
        #covariance = np.matmul(np.transpose(Z),Z)
        covariance = np.matmul(np.transpose(Z),Z)
        print(covariance.shape)
        # Z * Z.T and Z.T * Z has same eigenvalues for first N 
        self.calculateEigen(covariance, Z)
    def readDatabase(self):
        # opencv reads image in the reversed shape, you can think
        # max_h as the image height
        max_h = 0
        max_w = 0
        file_count = 0 
        os.chdir(self.eigenfaces_DB)
        for file in glob.glob("*.*"):
            file_count += 1
            im = cv2.imread(self.eigenfaces_DB + '/' + file)
            max_h = max(im.shape[0],max_h)
            max_w = max(im.shape[1],max_w)
            
        #first = cv2.imread(self.eigenfaces_DB + '/HR_11.tif')
        X  = np.zeros((file_count,max_h*max_w), dtype = 'uint8')
        file_count=0
        for file in glob.glob("*.*"):
            im = cv2.imread(self.eigenfaces_DB + '/' + file , cv2.IMREAD_GRAYSCALE)
            
            X[file_count,:] = np.array(im).reshape(max_h*max_w)
            file_count += 1
        return X
    def findAverageImage(self,X):
        # finds the mean of each pixel
        im,p = X.shape
        print(X.shape)
        avgIm = np.zeros((p), dtype='int32')
        
        for j in range(p):
            for i in range(im):
                avgIm[j] +=X[i,j]
            if(im != 0 ):
                avgIm[j] = avgIm[j] / im
        return avgIm
    
    def createDifferenceImages(self,avgIm,X):
        # substracts mean from each image
        im,p= X.shape
        Z = np.zeros((X.shape), dtype='uint8')
        for i in range(im):
            Z[i] = X[i] - avgIm
        # Z = N X 32
        return np.transpose(Z)
    
    def calculateEigen(self,cov,A):
        # cov = A.T  * A  --- 32x32 matrix
        
        # C = A * A.T --- NXN matrix
        start = timeit.default_timer()
        #C = np.matmul(A,A.T)
        eig,eiv = np.linalg.eig(cov)        
        eiv = eiv.T
        print(eiv)
        

        
        stop = timeit.default_timer()

        print('Time: ', stop - start)  

if __name__ == '__main__':
    App = QApplication(sys.argv)
    window = Window()
    cv2.destroyAllWindows()
    sys.exit(App.exec())
