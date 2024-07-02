import numpy as np
from tqdm import tqdm
x=np.array(([2,9],[1,5],[3,6]),dtype=float)
y=np.array(([92],[86],[89]),dtype=float)
x=x/np.amax(x,axis=0)
y=y/100
def sigmoid(x):
    return 1/(1+np.exp(-x))
def derivative(x):
    return x*(1-x)
epoch=5000
lr=10
input_neurons=2
hidden_neurons=3
output_neurons=1
wh=np.random.uniform(size=(input_neurons,hidden_neurons))
bh=np.random.uniform(size=(1,hidden_neurons))
wout=np.random.uniform(size=(hidden_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))
for i in range(epoch):
    hinpl=np.dot(x,wh)
    hinp=hinpl+bh
    hlayer_act=sigmoid(hinp)
    outinpl=np.dot(hlayer_act,wout)
    outinp=outinpl+bout
    output=sigmoid(outinp)
    E0=y-output
    outgrad=derivative(output)
    d_output=E0*outgrad
    EH=d_output.dot(wout.T)
    hiddengrad=derivative(hlayer_act)
    d_hiddenlayer=EH*hiddengrad
    wout+=hlayer_act.T.dot(d_output)*lr
    wh+=x.T.dot(d_hiddenlayer)*lr
print("Input: \n"+str(x))
print("Actual Output: \n"+str(y))
print("Predicted output: \n",output)

