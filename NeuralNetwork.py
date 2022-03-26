from sklearn.datasets import make_blobs, make_circles
from ANNUtils import *

#first exemple for NN 2 Layers
def execFirstExample():
    X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
    X = X.T
    y = y.reshape((1, y.shape[0]))

    print('dimensions de X:', X.shape)
    print('dimensions de y:', y.shape)

    plt.scatter(X[0, :], X[1, :], c=y, cmap='summer')
    plt.show()

    neural_network2(X, y, n1=32)

#execFirstExample()

#Cat and dogs example

from utilities import *

def execCatsDogs():
    
    X_train, y_train, X_test, y_test = load_data()

    print(X_train.shape)
    print(y_train.shape)
    print(np.unique(y_train, return_counts=True))

    print(X_test.shape)
    print(y_test.shape)
    print(np.unique(y_test, return_counts=True))

    plt.figure(figsize=(16, 8))
    for i in range(1, 10):
        plt.subplot(4, 5, i)
        plt.imshow(X_train[i], cmap='gray')
        plt.title(y_train[i])
        plt.tight_layout()
    plt.show()


    X_train_reshape = X_train.reshape(X_train.shape[0], -1) / X_train.max()
    X_train_reshape = X_train_reshape.T
    X_train_reshape = normalize( X_train_reshape )

    X_test_reshape = X_test.reshape(X_test.shape[0], -1) / X_test.max()
    X_test_reshape = X_test_reshape.T
    X_test_reshape = normalize( X_test_reshape )

    #X_train_reshape_norm = normalize( X_train_reshape)

    print("y-train shape = ", y_train.shape )
    y_train_reshape = y_train.T
    y_test_reshape = y_test.T

    m_train = 300
    m_test = 80
    #X_test_reshape = X_test_reshape[:,:m_test]
    #X_train_reshape = X_train_reshape[:,:m_train]
    #y_test_reshape = y_test_reshape[:,:m_test]
    #y_train_reshape = y_train_reshape[:,:m_train]


    print('dimensions de X_train_reshape:', X_train_reshape.shape)
    print('dimensions de y_train_reshape:', y_train_reshape.shape)
    print('dimensions de X_test_reshape:', X_test_reshape.shape)
    print('dimensions de y_test_reshape:', y_test_reshape.shape)

    #params = neural_network2(X_train_reshape, y_train_reshape, X_test_reshape, y_test_reshape, n1 = 128, learning_rate = 0.001, n_iter=80000, printTest = True)
    params = neural_network(X_train_reshape, y_train_reshape, X_test_reshape, y_test_reshape, neuronnesinternes = [ 16, 32, 32, 16 ], learning_rate = 0.05, n_iter=500000, printTest = True)

#execCatsDogs()


#Generic Neural Network
def testGenericNeuralNetwork():
    neuronnes = [ 2, 2, 2, 2, 2, 2, 32, 1]

    X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
    X = X.T
    y = y.reshape((1, y.shape[0]))

    print('dimensions de X:', X.shape)
    print('dimensions de y:', y.shape)

    parametres = initialisation( neuronnes )
    print( "parametres : ", parametres )

    activation = forward_propagation( X, parametres ) 
    print( "activation : ", activation )

    gradients = back_propagation( X, y, parametres, activation )
    print( "gradients : ", gradients )

    parametres = update( gradients, parametres, 0.01 )
    print( "parametres : ", parametres )

    reponse = predict( X, parametres )
    print( "reponse : ", reponse )

    neural_network(X, y, X_test = 0, y_test = 0, neuronnesinternes = [ 32, 16, 32, 16, 128 ], learning_rate = 0.1, n_iter = 10000, printTest = False)

#testGenericNeuralNetwork()


#MNIST handwritten Digits

from mnist import MNIST

mndata = MNIST('C:/Users/benoi/Desktop/FirstIA/Digits_data_set')

train_images, train_labels = mndata.load_training()

test_images, test_labels = mndata.load_testing()

X_train = np.array( train_images )
X_test = np.array( test_images )

y_train = np.array( train_labels )
y_test = np.array( test_labels )

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

print( X_train[ 1000 ] )
print ( y_train[ 1000 ] )

X_train_reshape = X_train.reshape(X_train.shape[0], -1) / X_train.max()
X_train_reshape = X_train_reshape.T

X_test_reshape = X_test.reshape(X_test.shape[0], -1) / X_test.max()
X_test_reshape = X_test_reshape.T

y_train_reshape = y_train.reshape(y_train.shape[0],1)
y_train_reshape = y_train_reshape.T
y_test_reshape = y_test.reshape(y_test.shape[0],1)
y_test_reshape = y_test_reshape.T

print(X_train_reshape.shape)
print(X_test_reshape.shape)
print(y_train_reshape.shape)
print(y_test_reshape.shape)

m_train = 300
m_test = 80
X_test_reshape = X_test_reshape[:,:m_test]
X_train_reshape = X_train_reshape[:,:m_train]
y_test_reshape = y_test_reshape[:,:m_test]
y_train_reshape = y_train_reshape[:,:m_train]

print(X_train_reshape.shape)
print(X_test_reshape.shape)
print(y_train_reshape.shape)
print(y_test_reshape.shape)

X_train_reshape = normalize( X_train_reshape )
X_test_reshape = normalize( X_test_reshape )

neural_network_digits(X_train_reshape, y_train_reshape, X_test_reshape, y_test_reshape, neuronnesinternes = [ 16 ], learning_rate = 0.1, n_iter = 10000, printTest = True)
