import librosa
import sys
import numpy as np


def simmx(A, B):
    [r1,c1] = A.shape
    [r2,c2] = B.shape

    if(r1 != r2 and r2 >= r1):
        if(r1 > r2):
            max = r1
        else:
            max = r2
    else:
        print("잘못 된 파일 입니다.")
        return -1;

    A1 = np.power(A, 2)
    EA = np.sqrt(A1.sum(axis = 0)) # 행렬

    B1 = np.power(B, 2)
    EB = np.sqrt(B1.sum(axis = 0))

    A = A.T
    A2 = np.zeros((c1,max))
    for i in range(0,c1):
        for j in range(0,r1):
            A2[i][j] = A[i][j]

    tmp = EA.reshape(c1,1)
    tmp = tmp * EB
    tmp2 = A2.dot(B)
    M = tmp2 / tmp

    return M

def dp(M):

    [r,c] = M.shape
    D = np.zeros((r+1, c+1))
    D_test = np.zeros((r+1,c+1))

    D[0][0] = 0
    for i in range(0,c+1):
        D[0][i] = sys.maxsize

    for i in range(0,r+1):
        D[i][0] = sys.maxsize


    for i in range(0,r):
        for j in range(0,c):
            D[1+i][1+j] = M[i][j]

    for i in range(0,r+1):
        for j in range(0,c+1):
            D_test[i][j] = D[i][j]



    #traceback
    phi = np.zeros((r,c))

    for i in range(1,r):
        for j in range(1,c):
            tmp = np.array([D[i][j], D[i][j + 1], D[i + 1][j]])
            dmax = tmp.min()
            dmin = tmp.max()
            tb = tmp.argmin()
            D_test[i + 1, j + 1] = D[i + 1, j + 1] + dmin
            D[i + 1, j + 1] = D[i + 1, j + 1] + dmax;
            phi[i][j] = tb;

    i = r;
    j = c;
    p = i;
    q = j;

    while i > 0 & j > 0:
        tb = phi[i][j]
        if(tb == 1):
            i = i-1
            j = j-1
        elif(tb == 2):
            i = i-1
        elif(tb == 3):
            j = j-1
        else:
            print("error")
            break
        p = [i,p]
        q = [j,q]

    D1 = np.zeros((r,c))

    for i in range(0,r):
        for j in range(0,c):
            D1[i][j] = D[1+i][1+j]

    return D_test,D1

def mfcc_def(y,s_r, x, x_r):
    A = librosa.feature.mfcc(y = y, sr = s_r)
    mfcc1 = np.array(A,float)
    mfcc1 = mfcc1.T

    B = librosa.feature.mfcc(y = x, sr = x_r)
    mfcc2 = np.array(B,float)
    mfcc2 = mfcc2.T
    M = simmx(mfcc1,mfcc2)
    if(M != -1):
        D_test,D1 = dp(1-M)
        res = 0

        if (D1[-1][-1] / D_test[-1][-1] == 0.0):
            res = 100
        else:
            res = round((D_test[-1][-1] - D1[-1][-1]) / D_test[-1][-1] * 1000)
            print("점수")
        print(res)

        return res

    return -1