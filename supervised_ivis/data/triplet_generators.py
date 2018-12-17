""" 
Triplet generators.

Functions for creating generators that will yield batches of triplets. 

"""

import sys
from scipy.sparse import issparse

import numpy as np
import threading


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return 

    def __next__(self): # Py3
        with self.lock:
            return next(self.it)

    def next(self): # Py2
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

def create_triplet_generator_from_labels(X, y, batch_size):
    return generate_triplets_from_labels(X, np.array(y), batch_size=batch_size)

@threadsafe_generator
def generate_triplets_from_labels(X, Y, batch_size=32):
    N_ROWS = X.shape[0]
    iterations = 0
    row_indexes = np.array(list(range(N_ROWS)), dtype=np.uint32)
    np.random.shuffle(row_indexes)

    placeholder_labels = np.array([0 for i in range(batch_size)])

    while True:
        triplet_batch = []
        
        for i in range(batch_size):
            if iterations >= N_ROWS:
                np.random.shuffle(row_indexes)
                iterations = 0
           
            triplet = triplet_from_labels(X, Y, row_indexes[iterations])            
            
            triplet_batch += triplet
            iterations += 1
        
        if (issparse(X)):
            triplet_batch = [[e.toarray()[0] for e in t] for t in triplet_batch]                 
            
        triplet_batch = np.array(triplet_batch)        
        yield ([triplet_batch[:,0], triplet_batch[:,1], triplet_batch[:,2]], placeholder_labels)

def triplet_from_labels(X, Y, index):
    """ A random (unweighted) positive example chosen. """
    N_ROWS = X.shape[0]
    triplets = []

    row_label = Y[index]
    neighbour_indexes = np.where(Y == row_label)[0]
    
    # Take a random neighbour as positive
    neighbour_ind = np.random.choice(neighbour_indexes)
    
    # Take a random non-neighbour as negative
    negative_ind = np.random.randint(0, N_ROWS)     # Pick a random index until one fits constraint. An optimization.
    while negative_ind in neighbour_indexes:
        negative_ind = np.random.randint(0, N_ROWS)
    
    triplets += [[X[index], X[neighbour_ind], X[negative_ind]]]
    return triplets

def create_triplets_from_positive_index_dict(X, positive_index_dict):
    N_ROWS = X.shape[0]
    triplets = []
    labels_placeholder = []
    for i in range(N_ROWS):
        try:
            for neighbour in positive_index_dict[i]:
                ind = i
                while ind == i or ind in positive_index_dict[i]:
                    ind = random.randrange(0, N_ROWS)
                triplets += [[X[i], X[neighbour], X[ind]]]
                labels_placeholder += [1]
        except:
            pass
    return np.array(triplets), np.array(labels_placeholder)

