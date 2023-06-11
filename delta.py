import math
import random
 
 #Sigmid derivative function activation function
def Sigmid(x):
    y = 1.0/ (1+math.exp(-x));
    return y;
 # W is the weight of the transmission weight parameter, X is the input training data transfer, D is the standard output parameter
def DeltaSGD (W,X,D):
         # Learning rate
    alpha = 0.9
    for  i in range(0,4):
        v = 0;
        d = D[i];
        for j in range(0,3):
            v=v+ W[j]*X[i][j];
                 # Obtain the output node
        y = Sigmid(v);
                 # Calculated error
        e = d - y;
        for j in range(0,3):
                         # Seek dw
            dw = y*(1-y)*e*alpha*X[i][j];
            W[j] = W[j]+ dw;
    return W;
 
if __name__ == '__main__':
         #Wij j represents the weight between the input node to the output node i
    W = []
         # Standard input
    data =[[0,0,1],[0,1,1],[1,0,1],[1,1,1]]
         # Standard output
    D =[0,0,1,1]
         # Initialize the weights
    for i in range(0,3):
        W.append(2*random.random()-1);
         # Training data
    for i in range(1,10001):
        W=DeltaSGD(W,data,D)
    y=0
         # Output the final training results
    for i in range(0,4):
        v =0;
        for j in range(0,3):
            v= v+ W[j]*data[i][j];
        y= Sigmid(v);
        print(y);
