from re import L
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm

def initialisation2(n0, n1, n2):

    W1 = np.random.randn(n1, n0)
    b1 = np.zeros((n1, 1))
    W2 = np.random.randn(n2, n1)
    b2 = np.zeros((n2, 1))

    parametres = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    return parametres

def initialisation( neuronnes ):
    parametresW = []
    parametresb = []
    for i in range( 1, len( neuronnes ) ):
        parametresW.append( np.random.randn( neuronnes[ i ], neuronnes[ i - 1 ] ) )
        parametresb.append( np.zeros( ( neuronnes[ i ], 1 ) ) )
    
    parametres = {
        'parametresW' : parametresW,
        'parametresb' : parametresb
    }

    return parametres

def forward_propagation2(X, parametres):

    W1 = parametres['W1']
    b1 = parametres['b1']
    W2 = parametres['W2']
    b2 = parametres['b2']

    Z1 = W1.dot(X) + b1
    A1 = 1 / (1 + np.exp(-Z1))

    Z2 = W2.dot(A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))

    activations = {
        'A1': A1,
        'A2': A2
    }

    return activations

def forward_propagation(X, parametres):

    W = parametres[ 'parametresW' ]
    b = parametres[ 'parametresb' ]

    activations = []

    Z1 = W[ 0 ].dot(X) + b[ 0 ]
    A1 = 1 / (1 + np.exp(-Z1))
    activations.append( A1 )

    for i in range( 1, len( W ) ):
        Z = W[ i ].dot( activations[ i - 1 ]) + b[ i ]
        A = 1 / (1 + np.exp(-Z))
        activations.append( A )

    return activations


def back_propagation2(X, y, parametres, activations):

    A1 = activations['A1']
    A2 = activations['A2']
    W2 = parametres['W2']

    m = y.shape[1]

    dZ2 = A2 - y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims = True)

    dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims = True)

    gradients = {
        'dW1' : dW1,
        'db1' : db1,
        'dW2' : dW2,
        'db2' : db2
    }
    
    return gradients

def back_propagation(X, y, parametres, activations):

    A = activations
    W = parametres['parametresW']

    m = y.shape[1]
    
    gradientsdW = []
    gradientsdb = []

    dZ = A[ len( W ) - 1 ] - y
    dW = 1 / m * dZ.dot(A[ len( W ) - 2 ].T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims = True)
    gradientsdW.insert(0, dW)
    gradientsdb.insert(0, db)

    for i in range( 1, len( W ) - 1 ):
        dZ = np.dot(W[ len( W ) - i ].T, dZ) * A[ len( W ) - i - 1 ] * ( 1 - A[ len( W ) - i - 1 ] )
        dW = 1 / m * dZ.dot(A[ len( W ) - i - 2].T)
        db = 1 / m * np.sum(dZ, axis=1, keepdims = True)
        gradientsdW.insert(0, dW)
        gradientsdb.insert(0, db)

    dZ = np.dot(W[ 1 ].T, dZ) * A[ 0 ] * ( 1 - A[ 0 ] )
    dW = 1 / m * dZ.dot(X.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims = True)
    gradientsdW.insert(0, dW)
    gradientsdb.insert(0, db)

    gradients = {
        'gradientsdW' : gradientsdW,
        'gradientsdb' : gradientsdb
    }
    
    return gradients

def back_propagation_digits(X, y, parametres, activations):

    A = activations
    W = parametres['parametresW']

    m = y.shape[1]
    
    gradientsdW = []
    gradientsdb = []

    dZ = A[ len( W ) - 1 ] - transformeVectorOfDigitsToMatrixOfVectors ( y[0] )
    dW = 1 / m * dZ.dot(A[ len( W ) - 2 ].T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims = True)
    gradientsdW.insert(0, dW)
    gradientsdb.insert(0, db)

    for i in range( 1, len( W ) - 1 ):
        dZ = np.dot(W[ len( W ) - i ].T, dZ) * A[ len( W ) - i - 1 ] * ( 1 - A[ len( W ) - i - 1 ] )
        dW = 1 / m * dZ.dot(A[ len( W ) - i - 2].T)
        db = 1 / m * np.sum(dZ, axis=1, keepdims = True)
        gradientsdW.insert(0, dW)
        gradientsdb.insert(0, db)

    dZ = np.dot(W[ 1 ].T, dZ) * A[ 0 ] * ( 1 - A[ 0 ] )
    dW = 1 / m * dZ.dot(X.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims = True)
    gradientsdW.insert(0, dW)
    gradientsdb.insert(0, db)

    gradients = {
        'gradientsdW' : gradientsdW,
        'gradientsdb' : gradientsdb
    }
    
    return gradients

def update2(gradients, parametres, learning_rate):

    W1 = parametres['W1']
    b1 = parametres['b1']
    W2 = parametres['W2']
    b2 = parametres['b2']

    dW1 = gradients['dW1']
    db1 = gradients['db1']
    dW2 = gradients['dW2']
    db2 = gradients['db2']

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parametres = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    return parametres

def update(gradients, parametres, learning_rate):
    W = parametres[ 'parametresW' ]
    b = parametres[ 'parametresb' ]
    dW = gradients[ 'gradientsdW' ]
    db = gradients[ 'gradientsdb' ]

    parametresW = []
    parametresb = []

    for i in range( 0, len( W ) ):
        Wn = W[ i ] - learning_rate * dW[ i ]
        bn = b[ i ] - learning_rate * db[ i ]
        parametresW.append( Wn )
        parametresb.append( bn )

    parametres = {
        'parametresW' : parametresW,
        'parametresb' : parametresb
    }

    return parametres

def predict2(X, parametres):
  activations = forward_propagation2(X, parametres)
  A2 = activations['A2']
  return A2 >= 0.5

def predict(X, parametres):
  activations = forward_propagation(X, parametres)
  Af = activations[ -1 ]
  return Af >= 0.5

def predict_digits(X, parametres):
    activations = forward_propagation(X, parametres)
    listPred = []
    listConfidencePred = []
    Af = activations[ -1 ]
    for i in range(len(Af[0])):
        max = 0
        pred = 0
        for j in range(10):
            if Af[j][i] > max:
                max = Af[j][i]
                pred = j
        listPred.append(pred)
        listConfidencePred.append( max )
    return ( np.array(listPred), np.array(listConfidencePred) )

def neural_network2(X, y, X_test = 0, y_test = 0, n1=32, learning_rate = 0.1, n_iter = 1000, printTest = False):

    # initialisation parametres
    n0 = X.shape[0]
    n2 = y.shape[0]
    np.random.seed(0)
    parametres = initialisation2(n0, n1, n2)

    train_loss = []
    train_acc = []
    history = []

    loss_test = []
    acc_test = []

    # gradient descent
    for i in tqdm(range(n_iter)):
        activations = forward_propagation2(X, parametres)
        gradients = back_propagation2(X, y, parametres, activations)
        parametres = update2(gradients, parametres, learning_rate)

        if i % 10 == 0:
            # Plot courbe d'apprentissage
            train_loss.append(log_loss(y.flatten(), activations['A2'].flatten()))
            y_pred = predict2(X, parametres)
            train_acc.append(accuracy_score(y.flatten(), y_pred.flatten()))

            #Test
            if printTest:
                activations_test = forward_propagation2( X_test, parametres )
                A2_test = activations_test[ "A2" ]
                loss_test.append( log_loss( y_test.flatten(), A2_test.flatten() ) )
                y_pred = predict2( X_test, parametres )
                acc_test.append( accuracy_score( y_test.flatten(), y_pred.flatten() ) )
            
            history.append([parametres.copy(), train_loss, train_acc, i])


    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='train loss')
    if printTest:
        plt.plot( loss_test, label = 'test loss' )
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='train acc')
    if printTest:
        plt.plot( acc_test, label = 'test acc' )
    plt.legend()
    plt.show()

    return parametres

def neural_network(X, y, X_test = 0, y_test = 0, neuronnesinternes = [ 32 ], learning_rate = 0.1, n_iter = 1000, printTest = False):

    # initialisation parametres
    n0 = X.shape[0]
    neuronnes = neuronnesinternes 
    neuronnes.insert( 0, n0 )
    n2 = y.shape[0]
    neuronnes.append( n2 )
    np.random.seed(0)
    parametres = initialisation( neuronnes )

    train_loss = []
    train_acc = []
    history = []

    loss_test = []
    acc_test = []

    # gradient descent
    for i in tqdm(range(n_iter)):
        activations = forward_propagation( X, parametres )
        gradients = back_propagation( X, y, parametres, activations )
        parametres = update( gradients, parametres, learning_rate )

        if i % 10 == 0:
            # Plot courbe d'apprentissage
            train_loss.append(log_loss(y.flatten(), activations[ -1 ].flatten()))
            y_pred = predict(X, parametres)
            train_acc.append(accuracy_score(y.flatten(), y_pred.flatten()))

            #Test
            if printTest:
                activations_test = forward_propagation( X_test, parametres )
                Af_test = activations_test[ -1 ]
                loss_test.append( log_loss( y_test.flatten(), Af_test.flatten() ) )
                y_pred = predict( X_test, parametres )
                acc_test.append( accuracy_score( y_test.flatten(), y_pred.flatten() ) )
            
            history.append([parametres.copy(), train_loss, train_acc, i])


    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='train loss')
    if printTest:
        plt.plot( loss_test, label = 'test loss' )
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='train acc')
    if printTest:
        plt.plot( acc_test, label = 'test acc' )
    plt.legend()
    plt.show()

    return parametres

def neural_network_digits(X, y, X_test = 0, y_test = 0, neuronnesinternes = [ 32 ], learning_rate = 0.1, n_iter = 1000, printTest = False):

    # initialisation parametres
    n0 = X.shape[0]
    neuronnes = neuronnesinternes 
    neuronnes.insert( 0, n0 )
    nf = 10
    neuronnes.append( nf )
    np.random.seed(0)
    parametres = initialisation( neuronnes )

    train_loss = []
    train_acc = []
    history = []

    loss_test = []
    acc_test = []

    # gradient descent
    for i in tqdm(range(n_iter)):
        activations = forward_propagation( X, parametres )
        gradients = back_propagation_digits( X, y, parametres, activations )
        parametres = update( gradients, parametres, learning_rate )

        if i % 10 == 0:
            # Plot courbe d'apprentissage
            #confidence = []
            #for j in range( y.shape[1]):
            #    test = y.flatten()
            #    index = test[j] - 1
            #    confidence.append( activations[ -1 ][ index ][ j ])
            #test = np.full((300,), [1])
            #test2 = np.array(confidence).flatten()
            #train_loss.append(log_loss(np.full((300,),[1]), np.array(confidence).flatten()))
            y_pred = predict_digits(X, parametres)[0]
            test = y.flatten()
            test2 = y_pred.flatten()
            train_acc.append(accuracy_score(y.flatten(), y_pred.flatten()))

            #Test
            if printTest:
                activations_test = forward_propagation( X_test, parametres )
                Af_test = activations_test[ -1 ]
                #confidence = []
                #for j in range( y_test.shape[1]):
                #    test = y_test.flatten()
                #    index = test[j] - 1
                #    confidence.append( Af_test[ index ][ j ])
                #loss_test.append( log_loss( np.full((300,), [1]), np.array( confidence ).flatten() ) )
                y_pred = predict_digits( X_test, parametres )[0]
                acc_test.append( accuracy_score( y_test.flatten(), y_pred.flatten() ) )
            
            history.append([parametres.copy(), train_loss, train_acc, i])


    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='train loss')
    if printTest:
        plt.plot( loss_test, label = 'test loss' )
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='train acc')
    if printTest:
        plt.plot( acc_test, label = 'test acc' )
    plt.legend()
    plt.show()

    return parametres


def normalize( X ):
    X_trainNorm = X.astype(np.double)
    for i in tqdm(range(X_trainNorm.shape[0])):
        for x in range(X_trainNorm.shape[1]):
            X_trainNorm[i][x] = X_trainNorm[i][x] / 255
    return X_trainNorm

import h5py
import numpy as np


def load_data():
    train_dataset = h5py.File('C:/Users/benoi/Desktop/FirstIA/datasets/trainset.hdf5', "r")
    X_train = np.array(train_dataset["X_train"][:]) # your train set features
    y_train = np.array(train_dataset["Y_train"][:]) # your train set labels

    test_dataset = h5py.File('C:/Users/benoi/Desktop/FirstIA/datasets/testset.hdf5', "r")
    X_test = np.array(test_dataset["X_test"][:]) # your train set features
    y_test = np.array(test_dataset["Y_test"][:]) # your train set labels
    
    return X_train, y_train, X_test, y_test

def transformDigitToVector ( n ):
    vector = np.zeros((10,1))
    vector[ n - 1 ] = 1
    return vector

def transformeVectorOfDigitsToMatrixOfVectors ( y ):
    matrix = np.zeros( ( 10, y.shape[0] ) )
    for i in range( len(y) ):
        matrix[ y[i] - 1 ][ i ] = 1
    return matrix