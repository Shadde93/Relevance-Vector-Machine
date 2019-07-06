import numpy as np
import random
import math
from sklearn import datasets, grid_search,cross_validation
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold

def generateData(noiseType,N):
    x = np.linspace(-10,10,N)
    t = (1/np.abs(x))*np.sin(np.abs(x))

    if noiseType == 'uniform':
        tNoise = np.random.uniform(-0.2,0.2,len(t))
    elif noiseType == 'gaussian':
        tNoise = np.random.normal(0,0.2,len(t))
        
    return x,t,tNoise
    
def generateDatas(N,F,choice):

    if(choice == 'f1'):
        X,Y = datasets.make_friedman1(N,F,noise = 1)
    elif(choice == 'f2'):
        X,Y = datasets.make_friedman2(N,F)
    elif(choice == 'f3'):
        X,Y == datasets.make_friedman3(N,F)
    elif(choice == 'boston'):
        boston = datasets.load_boston()
        X,Y = boston.data,boston.target

        
    return X,Y


def Kernel(x,y,choice,R):
    K = np.ones((len(x), len(x)))
    if(choice == 'spline'):
        for i in range(len(x)):
            for j in range(len(y)):
                K[i,j] = 1 + x[i]*y[j] + x[i]*y[j]*min(x[i],y[j])-((x[i]+x[j])/2)*(min(x[i],x[j])**2)+(min(x[i],x[j])**3)/3
    elif(choice == 'gaussian'):
        for i in range(len(x)):
            for j in range(len(y)):
                K[i,j] = math.exp(-(np.linalg.norm(x[i]-y[j])**2)/(R**2))
    return K

def phi(x,K):
    phi = np.ones((len(x),len(x) +1))
    for i in range(len(x)):
        for j in range(len(x)):
            phi[i,j] = K[i,j]
            phi[i,len(x)] = 1

    return phi

def Train(x,t,sigma,K):
    noiseVariance = sigma**2
    
    Phi = phi(x,K)

    alphas = np.ones((len(x)+1,1))
    phiTrans = np.transpose(Phi)
    mu = np.zeros((len(alphas),1))
    alphaThreshold = 10**12
    deltaThreshold = 0.0001

    
    for i in range(10000):
        useful = (alphas < alphaThreshold)
        mu[np.logical_not(useful)] = 0
        
        phiUsed = Phi[:,useful[:,0]]
        phiTransUsed = phiTrans [useful[:,0],:]
        
        cov = np.linalg.inv(np.diag(alphas[useful])+ (1/(noiseVariance)) * np.dot(phiTransUsed,phiUsed))
        

        previousAlpha = alphas[useful]
        
        gamma = 1 - np.multiply(alphas[useful],np.diag(cov))
        noiseVariance = (np.linalg.norm(t - np.dot(phiUsed,mu[useful]))**2)/(len(x)-sum(gamma))

        mu[useful] = (np.dot(cov,np.dot(phiTransUsed,t)))*(1/(noiseVariance)) 
        alphas[useful] = (gamma/(mu[useful]**2))
        
        delta = sum(np.abs(alphas[useful] - previousAlpha))
        if(delta < deltaThreshold):
            print('converged')
            break


    a = sum(useful) -1
    print("Number of relevance vectors" , a)
    indices = np.array(range(len(x)+1))[useful[:,0]]
    print("Estimated Noise Variance", np.sqrt(noiseVariance))
    return noiseVariance, mu, indices

def indicesNew(indices):
    indicesNew = indices[0:len(indices)-1]

    return indicesNew

        
def predictionsRV(mu, indices, x,K):
    kernelxStar = np.ones((len(x),len(x)+1))
    
    for i in range(len(x)):
        for j in range(len(x)):
             kernelxStar[i,j] = K[i,j]
             kernelxStar[i,len(x)] = 1

    yPred = np.dot(kernelxStar[:,indices],mu[indices])


    return yPred




def theSincFunctionPlots():

        
    x,y,yNoise = generateData('gaussian',100)
    K = Kernel(x,y,'spline',4)
    estimateSigma,mu,indices = Train(x,y,0.01,K)
    indiceN = indicesNew(indices)
    yPred = predictionsRV(mu,indices,x,K)
    y1 = y+yNoise


    plt.plot(x,y, 'b', label = 'trainingdata')
    plt.plot(x,yPred,'r+', label = 'true function')
    plt.plot(x[indiceN],y1[indiceN],'o')
    plt.plot(x,y+yNoise,".")
    mean_squared_error(y,yPred)

    plt.show()


def KfoldCrossValidation(X,Y):
    cv = KFold(n_splits = 5)
    cv.get_n_splits(X)

    KFold(n_splits = 5, random_state = None, shuffle = False)
    rms = np.zeros((5,5))
    j= 0

    for indexTrain,indexTest in cv.split(X):
        #print("Train:", indexTrain, "Test:", indexTest)
        XTrain,XTest = X[indexTrain],X[indexTest]
        YTrain,YTest = Y[indexTrain], Y[indexTest]
        print('----------------------------------')

        k = 5
        for i in range(0,5):
            k = k + 1
            K = Kernel(XTrain,XTrain,'gaussian',k)
            estimateSigma,mu,indices = Train(XTrain,YTrain,0.01,K)
            print('R = ', k)
            yPred = predictionsRV(mu,indices,XTrain,K)
            rms[i,j] = mean_squared_error(YTrain,yPred)
        j = j+1
    



    return yPred,mu,indices,estimateSigma,rms
    #for i in range(100):
##    XTrain,XTest,YTrain,YTest = generatedatas(80,6,'f1','5')
##    estimateSigma,mu,indices = Train(XTrain,YTrain,0.01)
##    yPred = predictionsRV(mu,indices,XTrain,)
##
##    plt.plot(XTrain,YTrain, '.', label = 'trainingdata')
##    plt.plot(XTrain,yPred,'r+', label = 'true function')
##    plt.plot(XTrain[indices],YTrain[indices],'o')

def ComputeRms(rms):
    Error = 0

    for i in range(0,5):
        for j in range(0,5):
            error = rms[i,j]
            Error += error

        aveError = Error/5    
        print(aveError)




def main():
    X,Y = generateDatas(300,10,'f1')
    yPred,mu,indices,estimateSigma,rms = KfoldCrossValidation(X,Y)
    #ComputeRms(rms)
    



theSincFunctionPlots()

#someBenchMarks(100)

#main()



