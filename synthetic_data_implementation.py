import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def datagenerate():
    x1=np.random.uniform(0, 1)
    x2=np.random.uniform(0,1)
    b=[-np.inf,-1,-0.1,0.25,1]
    e=np.random.normal(0,0.125)
    c=[]
    for i in range(5):
        if 10*((x1-0.5)*(x2-0.5)) + e >b[i]:
            c.append(i+1)
    return [x1,x2,max(c)]
data=[]
for i in range(100):
    data.append([])
    for j in range(7000):
        data[i].append(datagenerate)

%cd '/path../'
import algorithm
w1_synthetic=[]
loss11=[]
for i in range(100):
    x=np.transpose(np.transpose(data[i])[:-1])
    y=np.transpose(np.transpose(data[i])[-1])
    loss1,w1=algorithm.widrowhoff(x,y,7000,0.1)
    loss11.append(loss1)
    w1_synthetic.append(w1)
np.savetxt('/path../synthetic/loss1.csv',loss11, delimiter=',')
np.savetxt('/path../synthetic/w1_synthetic.csv',w1_synthetic, delimiter=',')

w2_synthetic=[]
b2_synthetic=[]
loss21=[]
for i in range(100):
    x=np.transpose(np.transpose(data[i])[:-1])
    y=np.transpose(np.transpose(data[i])[-1])
    loss2,w2,b2=algorithm.pranking(x,y,7000)
    loss21.append(loss2)
    w2_synthetic.append(w2)
    b2_synthetic.append(b2)

np.savetxt('/home/22n0457/synthetic/loss1.csv',loss11, delimiter=',')
np.savetxt('/home/22n0457/synthetic/w1_synthetic.csv',w1_synthetic, delimiter=',')

w2_synthetic=[]
b2_synthetic=[]
loss21=[]
#for i in range(100):
for i in range(1):
    x=np.transpose(np.transpose(data[i])[:-1])
    y=np.transpose(np.transpose(data[i])[-1])
    loss2,w2,b2=algorithm.pranking(x,y,7000)
    loss21.append(loss2)
    w2_synthetic.append(w2)
    b2_synthetic.append(b2)

np.savetxt('/home/22n0457/synthetic/loss2.csv',loss21, delimiter=',')
np.savetxt('/home/22n0457/synthetic/w2_synthetic.csv',w2_synthetic, delimiter=',')
np.savetxt('/home/22n0457/synthetic/b2_synthetic.csv',b2_synthetic, delimiter=',')

loss31=[]
for i in range(100):
    x=np.transpose(np.transpose(data[i])[:-1])
    y=np.transpose(np.transpose(data[i])[-1])
    loss3,m1=algorithm.uniformmulticlassalgo(x,y,7000)
    loss31.append(loss3)
    np.savetxt(f'/home/22n0457/synthetic/m1_synthetic___{i}.csv',m1, delimiter=',')
    
np.savetxt('/home/22n0457/synthetic/loss3.csv',loss31, delimiter=',')

loss41=[]
for i in range(100):
    x=np.transpose(np.transpose(data[i])[:-1])
    y=np.transpose(np.transpose(data[i])[-1])
    loss4,m2=algorithm.worstmulticlassalgo(x,y,7000)
    loss41.append(loss4)
    np.savetxt(f'/home/22n0457/synthetic/m2_synthetic___{i}.csv',m2, delimiter=',')

np.savetxt('/home/22n0457/synthetic/loss4.csv',loss41, delimiter=',')

loss51=[]
for i in range(100):
    x=np.transpose(np.transpose(data[i])[:-1])
    y=np.transpose(np.transpose(data[i])[-1])
    loss5,m3=algorithm.vimulticlassalgo(x,y,7000)
    loss51.append(loss5)
    np.savetxt(f'/home/22n0457/synthetic/m3_synthetic___{i}.csv',m3, delimiter=',')
    
np.savetxt('/home/22n0457/synthetic/loss5.csv',loss51, delimiter=',')

loss61=[]
for i in range(100):
    x=np.transpose(np.transpose(data[i])[:-1])
    y=np.transpose(np.transpose(data[i])[-1])
    loss6,m4=algorithm.mira(x,y,7000)
    loss61.append(loss6)
    np.savetxt(f'/home/22n0457/synthetic/m4_synthetic___{i}.csv',m4, delimiter=',')

np.savetxt('/home/22n0457/synthetic/loss6.csv',loss61, delimiter=',')