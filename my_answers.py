import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    """Generate input and out pairs for a given series
    and window size. Returns 2D array of input X and 2D array of label y
    @param: series  List of normalized data (eg: [0.3,0.5,0.11])
    @param: window_size size of the window (eg: 2)

    """
    # containers for input/output pairs
    # get the no.of rows to build X inputs and y output
    rows = len(series) - window_size 

    # generate empty 2D array input X of shape(rows, window_size)
    X = np.empty((rows, window_size))

    # label output y will always have only one label/output for a given input X
    # generate emtpy 2D array output y of shape(rows, window_size)
    y = np.empty((rows, 1))

    # construct input X and output y 
    # for X: build the sequence with window_size 
    # for y: get the label for a given sequence
    for i in range(0,rows):
        X[i] = series[i:i+window_size]
        y[i] = series[window_size+i]
    return X,y


# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    """Build the RNN regression model using LSTM and fully connected module. 
    Model architecture consists of two layers:
        1. First layer: Its an LSTM module with 5 hidden units and input_shape
                        of given window_size and step size
        2. Second layer: Its a fully connected Dense module with one unit
    
    Loss error: 'mean_squared_error' loss is used while performing regression

    @param step_size: step_size for each input sequence (eg: 2)
    @param window_size: window_size contains no.of input elements in a sequence (eg: 7)
    """
    # given - fix random seed - so we can all reproduce the same results on our default time series
    np.random.seed(0)

    # intialize the model 
    model = Sequential()

    # add 1st layer with 5 hidden input and define input_shape 
    model.add(LSTM(5, input_shape=(window_size,step_size)))

    # add 2nd layer fully connected with one unit
    model.add(Dense(1))

    # initialize optimizer and add it during regression
    optimizer = keras.optimizers.RMSprop(lr=0.001, epsilon=1e-08, decay=0.0)

    # use 'mean_squared_error' for the regression 
    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)


### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    """Cleaning the input text with non-english characters & numbers
    Identifies all unique characters in the input and check for non-english character
    & numbers. If found replaces non-english character & number with space. Also double space is replaced
    with single space

    @param text: contians input string (eg: sherlock holmes is awesome...)
    """
    # find all unique characters in the text - set will output unique characters
    text_unique = list(set(text))

    # remove as many non-english characters and character sequences as you can 
    # loop through each of the unique character
    for c in text_unique:
        # check if the character is non-english character
        # if so replace with ' ' 
        if not c.isalnum():
            text = text.replace(c, ' ')

    # shorten any extra dead space created above 
    # finally replace double space with singel space that might araise removal 
    # non english characters and numbers
    text = text.replace('  ',' ')


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    """Generate inputs X and output label y for a given window_size & step_size
    And returns inputs X and output y

    @params: text contians input string (eg: sherlock holmes is awesome...)
    @params: window_size no.of item sequence for each input of X (eg: 100)
    @params: step_size size for each input of X (eg: 5)

    inputs: returns as list
    outputs: returns as list
    """
    # containers for input/output pairs
    inputs = []
    outputs = []
    for i in range(0, len(text) - window_size, step_size):
        inputs.append(text[i:i+window_size])
        outputs.append(text[i+window_size])
    return inputs,outputs
