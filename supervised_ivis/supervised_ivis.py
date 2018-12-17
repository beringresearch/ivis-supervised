""" scikit-learn wrapper class for the Ivis algorithm. """

from .data.triplet_generators import create_triplet_generator_from_labels
from .nn.network import build_network, selu_base_network
from .nn.losses import triplet_loss

from keras.callbacks import EarlyStopping
from keras.models import load_model
from sklearn.base import BaseEstimator
import multiprocessing


class SupervisedIvis(BaseEstimator):
    """
    Ivis is a technique that uses an artificial neural network for dimensionality reduction, often useful for the purposes of visualization.  
    The network trains on triplets of data-points at a time and pulls positive points together, while pushing more distant points away from each other.  
    In this supervised version of the algorithm, triplets are sampled based on labels provided for each data point.

    Parameters
    ----------
    embedding_dims : int, optional (default: 2)
        Number of dimensions in the embedding space
    
    distance : string, optional (default: "pn")
        The loss function used to train the neural network. One of "pn", "euclidean", "softmax_ratio_pn", "softmax_ratio".
    
    batch_size : int, optional (default: 128)
        The size of mini-batches used during gradient descent while training the neural network. Must be less than the num_rows in the dataset.

    epochs : int, optional (default: 1000)
        The maximum number of epochs to train the model for. Each epoch the network will see a triplet based on each data-point once.

    n_epochs_without_progress : int, optional (default: 50)
        After n number of epochs without an improvement to the loss, terminate training early.

    margin : float, optional (default: 1)
        The distance that is enforced between points by the triplet loss functions

    model: keras.models.Model (default: None)
        The keras model to train using triplet loss. If provided, an embedding layer of size 'embedding_dims' will be appended to the end of the network. If not provided, a default 
        selu network composed of 3 dense layers of 128 neurons each will be created, followed by an embedding layer of size 'embedding_dims'.

    Attributes
    ----------
    model_ : keras Model 
        Stores the trained neural network model mapping inputs to embeddings
    
    loss_history_ : array-like, floats
        The loss history at the end of each epoch during training of the model.

    
    """                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     

    def __init__(self, embedding_dims=2, distance='pn', batch_size=128, epochs=1000, n_epochs_without_progress=50, margin=1, model=None):
        self.embedding_dims = embedding_dims
        self.distance = distance
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_epochs_without_progress = n_epochs_without_progress
        self.margin = margin
        self.model_ = model

    def _fit(self, X, y, val_x, val_y, shuffle_mode=True):
        datagen = create_triplet_generator_from_labels(X, y, batch_size=self.batch_size)

        val_datagen = None
        validation_steps = None
        loss_monitor = 'loss'
        
        if val_x is not None:
            val_datagen = create_triplet_generator_from_labels(X, y, batch_size=self.batch_size)
            validation_steps = int(val_x.shape[0] / self.batch_size)
            loss_monitor = 'val_loss'
        if self.model_:
            model = build_network(self.model_, embedding_dims=self.embedding_dims) 
        else:
            input_size = (X.shape[-1],)
            model = build_network(selu_base_network(input_size), embedding_dims=self.embedding_dims)

        try:
            model.compile(optimizer='adam', loss=triplet_loss(distance=self.distance, margin=self.margin))
        except KeyError:
            raise Exception('Loss function not implemented.')
        
        hist = model.fit_generator(datagen, 
            steps_per_epoch=int(X.shape[0] / self.batch_size), 
            epochs=self.epochs, 
            callbacks=[EarlyStopping(monitor=loss_monitor, patience=self.n_epochs_without_progress)],
            validation_data=val_datagen,
            validation_steps=validation_steps,
            shuffle=shuffle_mode,
            workers=multiprocessing.cpu_count() )
        self.loss_history_ = hist.history['loss']
        self.model_ = model.layers[3]

    def fit(self, X, y, val_x=None, val_y=None, shuffle_mode=True):
        self._fit(X, y, val_x, val_y, shuffle_mode)
        return self

    def fit_transform(self, X, y, val_x=None, val_y=None, shuffle_mode=True):
        self.fit(X, y, val_x, val_y, shuffle_mode)
        return self.transform(X)
        
    def transform(self, X):
        embedding = self.model_.predict(X)
        return embedding

    def save(self, filepath):
        self.model_.save(filepath)
    
    def load(self, filepath):
        model = load_model(filepath)
        self.model_ = model
        self.model_._make_predict_function()
        return self
