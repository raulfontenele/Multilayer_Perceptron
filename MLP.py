import numpy as np
import math

class MLP():
    def __init__(self,n_epoch,learn_rate,precision,hidden_layers):
        self.n_epoch = n_epoch
        self.learn_rate = learn_rate
        self.precision = precision
        self.hidden_layers = hidden_layers
        self.weights = []

    def fit(self,x_train,d_train):
        #Inicialização de variáveis da rede neural
        I,Y,gradient,self.weights,accumulator = self.init_variables(len(x_train[0,:]),len(d_train[0,:]))
        sqrt_error = [0, self.precision+1]
        epoch = 0

        while True:

            sqrt_error[0] = sqrt_error[1]
            

            for sample in range( len(x_train) ):
                # Foward
                for ctr in range( len(self.hidden_layers) ):
                    if ctr == 0:
                        I[ctr] = self.weight[ctr]*x_train[sample,:]
                        Y[ctr] = np.concatenate(-1, np.tanh(I[ctr]))
                    elif ctr == len(self.hidden_layers):
                        I[ctr] = self.weight[ctr]*I[ctr-1]
                        Y[ctr] = np.tanh(I[ctr])
                    else:
                        I[ctr] = self.weight[ctr]*I[ctr-1]
                        Y[ctr] = np.concatenate(-1, np.tanh(I[ctr]))

                #Backward
                for ctr in reversed( range( len(self.hidden_layers) ) ):
                    if ctr == 0:
                        gradient[ctr] = np.dot(self.weights[ctr+1][:,1::].transpose()*gradient[ctr + 1],) 
                        #adqMatrixTranspose((Matrix<double>)weigth[ctr + 1]) * (Matrix<double>)gradient[ctr + 1], (Matrix<double>)derTanHiperbolica((Matrix<double>)I[ctr]));
                        #accumulator[ctr] = (Matrix<double>)accumulator[ctr] + taxaAprendizado * (Matrix<double>)gradient[ctr] * vetEntrada;
                    elif ctr == len(self.hidden_layers):
                        I[ctr] = self.weight[ctr]*I[ctr-1]
                        Y[ctr] = math.tanh(I[ctr])
                    else:
                        I[ctr] = self.weight[ctr]*I[ctr-1]
                        Y[ctr] = np.concatenate(-1, math.tanh(I[ctr]))


            if(sqrt_error[1] - sqrt_error[0]<=self.precision or epoch>self.n_epoch):
                break

    def init_variables(self,x_train_length,d_train_length):
        weights = []
        I = []
        Y = []
        accumulator = []
        gradient = []

        # Inicializar as entradas e saídas de cada camada e o gradiente ( que possui as mesmas dimenssões da entrada da camada)
        for index in range(len(self.hidden_layers) +1):
            if index == len(self.hidden_layers):
                Iaux = np.zeros((d_train_length,1))
                Yaux = np.zeros((d_train_length,1))
            else:
                Iaux = np.zeros((self.hidden_layers[index],1))
                Yaux = np.zeros((self.hidden_layers[index]+1,1))

            I.append(Iaux)
            Y.append(Yaux)
            gradient(Iaux)

        # Inicializar as matrizes de pesos e o acumulador
        for index in range(len(self.hidden_layers)):
            if index == 0:
                weight = np.random.rand(self.hidden_layers[index], x_train_length)
                acc = np.zeros((self.hidden_layers[index], x_train_length))
            elif index == len(self.hidden_layers):
                weight = np.random.rand(self.hidden_layers[index-1], d_train_length)
                acc = np.zeros((self.hidden_layers[index-1], d_train_length))
            else:
                weight = np.random.rand(self.hidden_layers[index], self.hidden_layers[index-1])
                acc = np.zeros((self.hidden_layers[index], self.hidden_layers[index-1]))
            
            weights.append(weight)
            accumulator.append(acc)

        return I,Y,gradient,weights,accumulator

def diff_htan(value):
    return (1 - np.tanh(value)**2)



