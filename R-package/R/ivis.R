#' IVIS algorithm
#'
#' @param X numerical matrix to be reduced. Columns correspond to features.
#' @param y int class vector guiding supervised triplet selection.
#' @param embedding_dims int, optional (default: 2) Number of dimensions in the embedding space
#' @param distance string, optional (default: "pn")
#'        The loss function used to train the neural network. One of "pn", "euclidean", "softmax_ratio_pn", "softmax_ratio". 
#' @param batch_size int, optional (default: 128)
#'        The size of mini-batches used during gradient descent while training the neural network.
#' @param epochs int, optional (default: 1000)
#'        The maximum number of epochs to train the model for. Each epoch the network will see a triplet based on each data-point once.
#' @param n_epochs_without_progress int, optional (default: 50)
#'        After n number of epochs without an improvement to the loss, terminate training early.
#' @param margin float, optional (default: 1)
#'        The distance that is enforced between points by the triplet loss functions
#' @export

ivis <- function(X, y, embedding_dims = 2L,
    distance = "pn",
    batch_size = 128L,
    epochs = 1000L,
    n_epochs_without_progress = 50L,
    margin = 1){


    X <- data.matrix(X)
    if (!is.null(y)) y <- as.integer(y)
    embedding_dims <- as.integer(embedding_dims)
    batch_size <- as.integer(batch_size)
    epochs <- as.integer(epochs)
    n_epochs_without_progress = as.integer(n_epochs_without_progress)

    model <- ivis_object$Ivis(embedding_dims=embedding_dims,
        distance = distance, batch_size = batch_size,
        epochs = epochs, n_epochs_without_progress = n_epochs_without_progress,
        margin = margin)
    
    embeddings = model$fit_transform(X = X, y = y)
    return(embeddings)

    }
