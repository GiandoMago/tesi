import numpy as np
from torch.utils.data import DataLoader
from .strategy import Strategy
import pickle
from scipy.spatial.distance import cosine
import sys
import gc
from scipy.linalg import det
from scipy.linalg import pinv as inv
from copy import copy as copy
from copy import deepcopy as deepcopy
import torch
from torch import nn
import torchfile
from torch.autograd import Variable
import resnet
import vgg
import torch.optim as optim
import pdb
from torch.nn import functional as F
import argparse
import torch.nn as nn
from collections import OrderedDict
from scipy import stats
import numpy as np
import scipy.sparse as sp
from itertools import product
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.utils.extmath import row_norms, squared_norm, stable_cumsum
from sklearn.utils.sparsefuncs_fast import assign_rows_csr
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.utils.validation import _num_samples
from sklearn.utils import check_array
from sklearn.utils import gen_batches
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.metrics.pairwise import rbf_kernel as rbf
#from sklearn.externals.six import string_types
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances



class TesiDistance(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(TesiDistance, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.num_labels=len(np.unique(self.Y))
        

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        loader=DataLoader(self.handler(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled], transform=self.args['transformTest']),
                                shuffle=False, **self.args['loader_te_args'])
        data_x = []
        data_y=[]
        
        for x, y, _ in loader:
            data_x.append(x)
            data_y.append(y)

        x_final = torch.cat(data_x, dim=0)
        y_final=torch.cat(data_y,dim=0)
        chosen = self.init_centers_mine(x_final, n)
        #chosen = self.init_centers_mine(x_final[-5:], n) #Da provare con solo 5 unlabeled
        
        return idxs_unlabeled[chosen]
    
    # Metodo per inizializzare i centroidi utilizzando una variante personalizzata dell'algoritmo K-means

    def init_centers_mine(self, X, K):       

        
        # Calcola la norma Euclidea per ogni input in X lungo gli assi (1, 2, 3)
        norms = torch.norm(X, p=2, dim=(1, 2, 3))
        
        # Trova l'indice dell'input con la norma massima
        max_norm_index = torch.argmax(norms)
        
        # Inizializza il primo centroide con l'input corrispondente alla norma massima
        mu = [X[max_norm_index]]
        
        # Lista per memorizzare gli indici di tutti gli input selezionati come centroidi
        indsAll = [max_norm_index]
        
        # Lista per memorizzare l'indice del centroide a cui ogni input è assegnato
        centInds = [0.] * len(X)

        # Variabile di conteggio per il numero di centri aggiunti
        cent = 0
        
        # Contatore per debugging
        cont = 0
        
        while len(mu) < K:
            
            cont += 1
            print("nell'inizializzazione dei centri sono al vettore: ",cont)

            if len(mu) == 1:
                # Calcola la distanza tra il primo centroide e tutti gli input
                #D2 = torch.cdist(mu[-1].view(1, -1), X)
                D2 = self.calculateDistance(mu[-1], X)
            else:
                # Calcola la distanza tra il primo centroide e tutti gli input
                newD = self.calculateDistance(mu[-1], X)
            

                
                # Aggiorna la distanza solo se è più piccola di quella attualmente assegnata
                for i in range(len(X)):
                    if D2[i] > newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            
            # Verifica se tutte le distanze sono zero, in quel caso effettua una pausa per il debugging
            if torch.sum(D2).item() == 0.0:
                pdb.set_trace()
            
            # Converte la distanza in un tensore float e lo riformatta
            D2 = D2.float().view(-1)
            
            # Calcola le probabilità di selezionare un indice basate sulla distanza
            Ddist = (D2 ** 2) / torch.sum(D2 ** 2)
            
            # Crea una distribuzione discreta personalizzata basata sulle probabilità
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist.numpy()))
            
            # Genera un indice casuale utilizzando la distribuzione personalizzata, evitando gli indici già selezionati
            ind = customDist.rvs(size=1)[0]
            while ind in indsAll:
                ind = customDist.rvs(size=1)[0]
            
            # Aggiunge l'input selezionato come nuovo centroide
            mu.append(X[ind])
            
            # Aggiunge l'indice all'elenco degli indici di tutti gli input selezionati come centroidi
            indsAll.append(ind)
            
            # Aumenta il numero del centroide
            cent += 1
        
        # Restituisce la lista degli indici degli input selezionati come centroidi
        return indsAll



    # Metodo per calcolare le distanze tra un input e una lista di input utilizzando gradienti e probabilità

    def calculateDistance(self, x, X):
        # Ottiene la lunghezza della lista di input X
        m = len(X)
        
        # Crea un tensore di zeri delle dimensioni m per memorizzare le distanze
        distances = torch.zeros(m)
        
        # Itera su ogni input nella lista X
        for j in range(m):
            # Inizializza la distanza per l'input j
            dist = 0.0
            
            # Itera su ogni classe per calcolare i gradienti e le probabilità
            for l1 in range(self.num_labels):
                # Ottiene il gradiente e la probabilità dell'input x rispetto alla classe l1
                x_grad = self.get_grad(x, l1)
                x_prob = self.get_prob(x, l1)
                
                # Ottiene il gradiente e la probabilità dell'input X[j] rispetto a ogni classe l2
                for l2 in range(self.num_labels):
                    y_grad = self.get_grad(X[j], l2)
                    y_prob = self.get_prob(X[j], l2)
                    
                    # Calcola la distanza tra i gradienti moltiplicata per le probabilità
                    dist += torch.norm(x_grad - y_grad) * x_prob * y_prob
            
            # Memorizza la distanza calcolata per l'input j
            distances[j] = dist
        
        # Restituisce il tensore delle distanze
        return distances

