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
    def vectorize(self,X,i,arr,h,w):
        X[i,:] = np.array(arr).reshape(h*w)
        return X
    def readDatabase(self,DB_name,vectorize = False):
        # opencv reads image in the reversed shape, you can think
        # max_h as the image height
        max_h = 0
        max_w = 0
        file_count = 0 
        os.chdir(DB_name)
        for file in glob.glob("*.*"):
            file_count += 1
            im = cv2.imread(DB_name + '/' + file)
            max_h = max(im.shape[0],max_h)
            max_w = max(im.shape[1],max_w)
            
        #first = cv2.imread(self.eigenfaces_DB + '/HR_11.tif')
        if(vectorize):
            X  = np.zeros((file_count,max_h*max_w), dtype = 'uint8')
        else:
            X  = np.zeros((file_count,max_h,max_w), dtype = 'uint8')    
        file_count=0
        for file in glob.glob("*.*"):
            im = cv2.imread(DB_name + '/' + file , cv2.IMREAD_GRAYSCALE)
            if(vectorize):
                X = self.vectorize(X,file_count,im,max_h,max_w)
            else:
                X[file_count,:,:] = np.array(im)
            file_count += 1
        return X
    
    def opticalFlowAction(self):
        self.importInput(0)
        X = self.readDatabase(self.optic_DB,vectorize = False)
        
        B = self.findBackgroundImage(X)
        cv2.imwrite('../bg.png' , B)
        A   = np.zeros((2500,2), dtype='float64')
        b   = np.zeros((2500,1), dtype='float64')
        vel = np.zeros((5,3,2), dtype='float64')
        T,h,w = X.shape
        for t in range(T-1):
            start = timeit.default_timer()
            for m in range(0,5):
                for n in range(0,3):
                    
                    for ii in range(0,50):
                        i = (50*n) + ii
                        for jj in range(0,50):
                            
                            
                            j = (50*m) + jj
                            
                            
                            
                            Ix,Iy,It = self.calculateGradient(X,i,j,t)
                            A[((ii-1) * (jj-1)) + jj -1] = [Ix,Iy]
                            b[((ii-1) * (jj-1)) + jj -1] = It
                                                    
                    velocities = np.matmul((-1 * np.linalg.inv(np.matmul(A.T,A))), np.matmul(A.T,b))
                    if (t == 38):
                        vel[m,n,0] = velocities[0,0]
                        vel[m,n,1] = velocities[1,0]
                            
            stop = timeit.default_timer()
        self.drawArrows(X[38,:,:],vel)
            #print('Time: ', stop - start) 
            #print(velocities)
    def drawArrows(self,im,vel):
        h,w = im.shape
        arrowedIm = np.zeros((h,w,3), dtype = 'uint8')
        arrowedIm[:,:,0] = im
        arrowedIm[:,:,1] = im
        arrowedIm[:,:,2] = im
        
        maxHVel = -50
        maxVVel = -50
        minVVel = 50
        minHVel = 50
        for i in range(5):
            for j in range(3):
                maxHVel = max(maxHVel, vel[i,j,1])
                maxVVel = max(maxVVel, vel[i,j,0])
                minHVel = min(minHVel, vel[i,j,1])
                minVVel = min(minVVel, vel[i,j,0])
        
        if(maxHVel-minHVel != 0):
            hVel = int(25/(maxHVel-minHVel))
        
        if(maxVVel-minVVel != 0):
            vVel = int(25/(maxVVel-minVVel))
            
        
        
        print("Max H Velocity " + str(maxHVel))
        print("Max V Velocity " + str(maxVVel))
        
        print("Min H Velocity " + str(minHVel))
        print("Min V Velocity " + str(minVVel))    
        
        print("h interval " + str(hVel))
        print("v interval " + str(vVel))
        for i in range(5):
            for j in range(3):
                print(str(i * 50)+ " " +str(j * 50)+  " v velocity " + str(vel[i,j,0]))
                print(str(i * 50) +" " + str(j * 50)+ " h velocity " + str(vel[i,j,1]))
                p1 = (0,0)
                p2 = (0,0)
                y1 = int((i*50)+25 + (vVel * (vel[i,j,0]- minVVel)))
                y2 = int((i*50)+25 - (vVel * (vel[i,j,0]- minVVel)))
                x1 = int((j*50)+25 + (hVel * (vel[i,j,1]- minHVel)))
                x2 = int((j*50)+25 - (hVel * (vel[i,j,1]- minHVel)))
                if(vel[i,j,1] > 0 and vel[i,j,0] > 0):
                    p1 = (y2,x2)
                    p2 = (y1,x1)
                elif(vel[i,j,1] <= 0 and vel[i,j,0] > 0):
                    p1 = (y1,x2)
                    p2 = (y2,x1)
                elif(vel[i,j,1] > 0 and vel[i,j,0] <= 0):
                    p1 = (y2,x1)
                    p2 = (y1,x2)
                else:
                    p1 = (y1,x1)
                    p2 = (y2,x2)
                cv2.arrowedLine(arrowedIm,p1,p2,(0,255,0) ,1)
                
        cv2.imwrite('../arrowed.png',arrowedIm )
    def calculateGradient(self,X,i,j,t):
#        Ix = ((X[t,i+1,j] + X[t+1,i+1,j] + X[t,i+1,j+1] + X[t+1,i+1,j+1]) - (X[t,i,j] + X[t+1,i,j] + X[t,i,j+1] + X[t+1,i,j+1]))/4
#        Iy = ((X[t,i,j+1] + X[t+1,i,j+1] + X[t,i+1,j+1] + X[t+1,i+1,j+1]) - (X[t,i,j] + X[t+1,i,j] + X[t,i+1,j] + X[t+1,i+1,j]))/4
#        It = ((X[t+1,i,j] + X[t+1,i,j+1] + X[t+1,i+1,j] + X[t+1,i+1,j+1]) - (X[t,i,j] + X[t,i+1,j] + X[t,i,j+1] + X[t,i+1,j+1]))/4
        Ix = (( int(X[t,i+1,j]) + int(X[t+1,i+1,j]) + int(X[t,i+1,j+1]) + int(X[t+1,i+1,j+1])) - (int(X[t,i,j]) + int(X[t+1,i,j]) + int(X[t,i,j+1]) + int(X[t+1,i,j+1])))/4
        Iy = ((int(X[t,i,j+1]) + int(X[t+1,i,j+1]) + int(X[t,i+1,j+1]) + int(X[t+1,i+1,j+1])) - (int(X[t,i,j]) + int(X[t+1,i,j]) + int(X[t,i+1,j]) + int(X[t+1,i+1,j])))/4
        It = ((int(X[t+1,i,j]) + int(X[t+1,i,j+1]) + int(X[t+1,i+1,j]) + int(X[t+1,i+1,j+1])) - (int(X[t,i,j]) + int(X[t,i+1,j]) + int(X[t,i,j+1]) + int(X[t,i+1,j+1])))/4
        return Ix,Iy,It
        
    def findBackgroundImage(self,X):
        t,h,w = X.shape
        B = np.zeros((h,w), dtype = 'int32')
        for i in range(t):
            for x in range(h):
                for y in range(w):
                    B[x,y] += X[i,x,y]
        
        if (t != 0):
            B = B / t
        return B
    def eigenFacesAction(self):
        self.importInput(1)
        X = self.readDatabase(self.eigenfaces_DB,vectorize = True)
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
