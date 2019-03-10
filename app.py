####################################################################################################
#
## Author: Marcel van den Bosch
## Date: 2019-03-10
## Email: marcel.vandenbosch@atos.net
#
## Description: Dash/Flask demo app code for the Deep Learning session at the Innovatos 2019 event.
####################################################################################################

# standard library
import os
import shutil

# dash libs
import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.tools as tls
import flask
import base64
import io
from io import StringIO
import sys
import traceback

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


## The Tensorflow stuff
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import backend as k
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras.callbacks
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop, Adam
import tensorflow.keras.callbacks
from tensorflow.keras.callbacks import LambdaCallback
import pickle
import random


## Set dash app settings
server = flask.Flask('app')
server.secret_key = os.environ.get('secret_key', 'secret')

app = dash.Dash('app', server=server,external_stylesheets=[dbc.themes.COSMO])
app.scripts.config.serve_locally = True
app.config.supress_callback_exceptions = True



#Below are to be globals (so they can be used across flask calls) - Initialize empty
source_state = 1;
train_cat_fnames = []
train_dog_fnames = []
train_cats_dir = ''
train_dogs_dir = ''
validation_cats_dir = ''
validation_dogs_dir = ''
train_dir = ''
validation_dir = ''
train_datagen = None
test_datagen = None
train_generator = None
validation_generator = None
model = None
img_input = None
history = None
training_log = 'Not started.'
training_state = 0;
source_data_classes = None


# Helper function to 'un'-backspace the output from Keras
def newString(S):
    q = []
    for i in range(0, len(S)):
        if S[i] != '\x08':
            q.append(S[i])
        elif len(q) != 0:
            q.pop()

    # Build final string
    ans = ''

    while len(q) != 0:
        ans += q[0]
        q.pop(0)

    # return final string
    return ans


## Layout code for the nav header
def makeHeaderLayout():
    output = dbc.Card(
        [
            dbc.CardHeader(
                dbc.Tabs(
                    [
                        dbc.Tab(label="Welcome", tab_id="tab-welcome"),
                        dbc.Tab(label="Load Data", tab_id="tab-sourcedata"),
                        dbc.Tab(label="Explore", tab_id="tab-exploredata"),
                        dbc.Tab(label="Model Summary", tab_id="tab-modelsummary"),
                        dbc.Tab(label="Training", tab_id="tab-training"),
                        dbc.Tab(label="Results", tab_id="tab-results"),
                        dbc.Tab(label="Deep Layers", tab_id="tab-layers"),
                        ##dbc.Tab(label="Google Cloud ML Engine", tab_id="tab-cloudmle"),  ## Todo: This part of the demo is not yet implemented
                    ],
                    id="card-tabs",
                    card=True,
                    active_tab="tab-welcome",
                    className="mt-3",
                ), className='text-white bg-dark mb-3'  # className="mt-3"
            ),
            dbc.CardBody(dbc.CardText(id="card-content")),
        ], style={'width': '100%', 'margin': '0 auto'}
    )

    return output


#########################
# Dashboard Layout / View
#########################

# Set up Dashboard and create layout
app.layout = html.Div([
    # Page Header
    html.Div([
        html.Div([

            html.Div([

                html.Div([
                    # html.Img(src='/images/atos-logo-menu-bar.png'), html.H1('Project Header'),
                    html.Div([html.Img(src='/images/atos-logo-menu-bar.png', height='45px')],
                             style={'float': 'left', 'padding': '25px'}),
                    html.Div([html.H2('Innovatos 2019'), html.B('Deeplearning Demo')],
                             style={'padding': '15px', 'float': 'right'}),
                ], style={'top': '1%', 'left': '25%', 'position': 'absolute', 'width': '100%', 'min-width': '450px',
                          'max-width': '650px', 'height': 'auto', 'vertical-align': 'middle',
                          'z-index': '1', 'padding': '0', 'background-color': 'white'}),
                html.Div([html.Button('Reset', id='btn-reset', className='btn btn-warning', n_clicks=0),
                          html.Div(id='container-button-reset'), html.Br([]),
                          html.Button('Load', id='btn-load-model', className='btn btn-primary'),
                          html.Div(id='container-button-load')
                          ], style={'float': 'right', 'text-align': 'right'})
            ], style={'height': '150px'})],
            style={'background-image': 'url(/images/page_header_default.png)', 'background-repeat': 'repeat-x',
                   'width': '100%', 'margin': '0 auto'}),
        html.Div(style={'height': '25px'})
        , html.Div(id='navheader', children=[
            makeHeaderLayout()])
    ], style={'width': '75%', 'margin': '0 auto'})
], style={'background-image': 'url(/images/bg_pre-header_shadow.png)', 'background-repeat': 'repeat-x',
          'font-family': '"Stag Sans Book",Verdana,Arial,Helvetica,sans-serif',
          'font-size': '0.9rem'})

def resetAppState():
    global train_cat_fnames
    global train_dog_fnames
    global train_cats_dir
    global train_dogs_dir
    global validation_cats_dir
    global validation_dogs_dir
    global source_state
    global train_dir
    global validation_dir
    global model
    global img_input
    global history
    global training_log
    global train_generator
    global validation_generator
    global training_state;
    global source_data_classes

    training_state = 1;
    source_state = 1;
    train_cat_fnames = []
    train_dog_fnames = []
    train_cats_dir = ''
    train_dogs_dir = ''
    validation_cats_dir = ''
    validation_dogs_dir = ''
    train_dir = ''
    validation_dir = ''
    train_datagen = None
    test_datagen = None
    train_generator = None
    validation_generator = None
    model = None
    img_input = None
    history = None
    training_log = 'Not started.'
    source_data_classes = []


def listDataSet(datadir='cats_and_dogs_filtered', classA='cats', classB='dogs'):
    global train_cat_fnames
    global train_dog_fnames
    global train_cats_dir
    global train_dogs_dir
    global validation_cats_dir
    global validation_dogs_dir
    global train_dir
    global validation_dir

    base_dir = os.path.join('./data/', datadir)

    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    # Directory with our training cat pictures
    train_cats_dir = os.path.join(train_dir, classA)
    # Directory with our training dog pictures
    train_dogs_dir = os.path.join(train_dir, classB)

    # Directory with our validation cat pictures
    validation_cats_dir = os.path.join(validation_dir, classA)

    # Directory with our validation dog pictures
    validation_dogs_dir = os.path.join(validation_dir, classB)

    train_cat_fnames = os.listdir(train_cats_dir)
    train_cat_fnames.sort()
    train_dog_fnames = os.listdir(train_dogs_dir)
    train_dog_fnames.sort()

    return True


def createGenerators(labelClasses):
    global train_datagen
    global test_datagen

    global source_state
    global train_cat_fnames
    global train_dog_fnames
    global train_cats_dir
    global train_dogs_dir
    global validation_cats_dir
    global validation_dogs_dir
    global train_generator
    global validation_generator

    # All images will be rescaled by 1./255
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Flow training images in batches of 20 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=20,
        classes=labelClasses,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

    # Flow validation images in batches of 20 using test_datagen generator
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        classes=labelClasses,
        class_mode='binary')


def getSourceDataExampleImg():

    # Parameters for our graph; we'll output images in a 4x4 configuration
    nrows = 4
    ncols = 4

    # Index for iterating over images
    pic_index = 0

    # Set up matplotlib fig, and size it to fit 4x4 pics
    fig = plt.gcf()
    fig.set_size_inches(ncols * 4, nrows * 4)

    pic_index += 8
    next_cat_pix = [os.path.join(train_cats_dir, fname)
                    for fname in train_cat_fnames[pic_index - 8:pic_index]]
    next_dog_pix = [os.path.join(train_dogs_dir, fname)
                    for fname in train_dog_fnames[pic_index - 8:pic_index]]

    for i, img_path in enumerate(next_cat_pix + next_dog_pix):
        # Set up subplot; subplot indices start at 1
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off')  # Don't show axes (or gridlines)

        img = mpimg.imread(img_path)
        plt.imshow(img)

    return fig


def getLossPlot(train_history):
    fig = plt.figure()

    # Retrieve a list of accuracy results on training and test data
    # sets for each training epoch
    acc = train_history.history['acc']
    val_acc = train_history.history['val_acc']

    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']

    # Get number of epochs
    epochs = range(len(acc))

    ax1 = fig.add_subplot(221)
    # Plot training and validation accuracy per epoch
    ax1.plot(epochs, acc)
    ax1.plot(epochs, val_acc)
    plt.title('Training and validation accuracy')

    # Plot training and validation loss per epoch
    ax2 = fig.add_subplot(222)
    ax2.plot(epochs, loss)
    ax2.plot(epochs, val_loss)
    plt.title('Training and validation loss')

    return fig


def getModel():
    # Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
    # the three color channels: R, G, and B
    img_input = layers.Input(shape=(150, 150, 3))

    # First convolution extracts 16 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    x = layers.Conv2D(16, 3, activation='relu')(img_input)
    x = layers.MaxPooling2D(2)(x)

    # Second convolution extracts 32 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)

    # Third convolution extracts 64 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)

    # Flatten feature map to a 1-dim tensor so we can add fully connected layers
    x = layers.Flatten()(x)

    # Create a fully connected layer with ReLU activation and 512 hidden units
    x = layers.Dense(512, activation='relu')(x)

    # Create output layer with a single node and sigmoid activation
    output = layers.Dense(1, activation='sigmoid')(x)

    # Create model:
    # input = input feature map
    # output = input feature map + stacked convolution/maxpooling layers + fully
    # connected layer + sigmoid output layer
    model = Model(img_input, output)
    return ([model, img_input])


def addTrainingLog(msg):
    global training_log
    import datetime as datetime
    training_log = '[' + str(datetime.datetime.now()) + ']  ' + msg + '\r\n' + training_log


def logBatch(batch, msg, every_n=10):
    if batch % every_n == 0:
        addTrainingLog(msg)


def StartTraining(learning_rate=0.001, optimizer='RMSprop', epochs=3, steps_per_epoch=100, validation_steps=50):
    global train_datagen
    global test_datagen
    global model
    global img_input
    global history
    global training_log;
    global train_generator
    global validation_generator
    global training_state

    model, img_input = getModel()

    traceback.print_stack(file=sys.stdout)

    training_log = "Training model started!"

    training_state = 1

    if optimizer == 'Adam':
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=learning_rate),
                      metrics=['acc'])
    else:
        model.compile(loss='binary_crossentropy',
                      optimizer=RMSprop(lr=learning_rate),
                      metrics=['acc'])

    addTrainingLog("Model compiled")

    epoch_print_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: addTrainingLog(
        "{Epoch: %i} %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in logs.items()))))
    batch_print_callback = LambdaCallback(on_batch_end=lambda batch, logs: logBatch(batch, "{Batch: %i} %s" % (
    batch, ", ".join("%s: %f" % (k, v) for k, v in logs.items()))))

    callbacks = [epoch_print_callback, batch_print_callback]

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,  # 2000 images = batch_size * steps
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,  # 1000 images = batch_size * steps
        verbose=0,
        callbacks=callbacks)

    addTrainingLog("Model training is completed")
    model.save('export/model_trained.h5')

    pickle_out = open("./export/model_trained_history.pickle", "wb")
    del history.model  # otherwise we cannot pickle it
    pickle.dump(history, pickle_out)
    pickle_out.close()

    addTrainingLog("Model saved: ./export/model_trained.h5 | History saved: ./export/model_trained_history.pickle ")

    training_state = 2;

    return True


def predictOnRandomTestImage(myModel):
    import numpy as np
    import random

    global validation_cats_dir
    global validation_dogs_dir

    # Let's prepare a random input image of a cat or dog from the training set.
    validation_cat_fnames = os.listdir(validation_cats_dir)
    validation_cat_fnames.sort()
    validation_dog_fnames = os.listdir(validation_dogs_dir)
    validation_dog_fnames.sort()

    cat_img_files = [os.path.join(validation_cats_dir, f) for f in validation_cat_fnames]
    dog_img_files = [os.path.join(validation_dogs_dir, f) for f in validation_dog_fnames]
    img_path = random.choice(cat_img_files + dog_img_files)

    img = load_img(img_path, target_size=(150, 150))  # this is a PIL image
    x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
    x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

    # Rescale by 1/255
    x /= 255

    # Let's run our image through our network, thus obtaining all
    # intermediate representations for this image.
    result = myModel.predict(x)

    return ([img_path, result])


def plotDeepLayers():
    import numpy as np
    import random
    from tensorflow.keras.preprocessing.image import img_to_array, load_img
    from tensorflow.keras.models import load_model
    import matplotlib.pyplot as plt
    import io
    import base64

    global model
    global img_input
    global train_cats_dir
    global train_dogs_dir

    # model, img_input = getModel() -- blank model
    model = load_model("./export/model_trained.h5")
    img_input = model.inputs

    # Let's define a new Model that will take an image as input, and will output
    # intermediate representations for all layers in the previous model after
    # the first.
    successive_outputs = [layer.output for layer in model.layers[1:]]
    visualization_model = Model(img_input, successive_outputs)

    # Let's prepare a random input image of a cat or dog from the training set.
    cat_img_files = [os.path.join(train_cats_dir, f) for f in train_cat_fnames]
    dog_img_files = [os.path.join(train_dogs_dir, f) for f in train_dog_fnames]
    img_path = random.choice(cat_img_files + dog_img_files)

    img = load_img(img_path, target_size=(150, 150))  # this is a PIL image
    x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
    x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

    # Rescale by 1/255
    x /= 255

    # Let's run our image through our network, thus obtaining all
    # intermediate representations for this image.
    successive_feature_maps = visualization_model.predict(x)

    # These are the names of the layers, so can have them as part of our plot
    layer_names = [layer.name for layer in model.layers]

    plot_collection = []

    # Now let's display our representations
    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
        if len(feature_map.shape) == 4:
            fig = plt.figure()
            # Just do this for the conv / maxpool layers, not the fully-connected layers
            n_features = feature_map.shape[-1]  # number of features in feature map
            # The feature map has shape (1, size, size, n_features)
            size = feature_map.shape[1]
            # We will tile our images in this matrix
            display_grid = np.zeros((size, size * n_features))
            for i in range(n_features):
                # Postprocess the feature to make it visually palatable
                x = feature_map[0, :, :, i]
                x -= x.mean()
                x /= x.std()
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype('uint8')
                # We'll tile each filter into this big horizontal grid
                display_grid[:, i * size: (i + 1) * size] = x
            # Display the grid
            scale = 20. / n_features
            plt.figure(figsize=(scale * n_features, scale))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')

            buffer_ = io.BytesIO()
            plt.savefig(buffer_, format="png")
            buffer_.seek(0)
            encoded_image = base64.b64encode(buffer_.getvalue()).decode()

            plot_collection.append(encoded_image)

    return plot_collection

### UI Code here for each tab
def getContentSourceData():
    global source_state
    global train_cat_fnames
    global train_dog_fnames
    global train_cats_dir
    global train_dogs_dir
    global validation_cats_dir
    global validation_dogs_dir

    output = html.Div(id='tab-datasource-contents', children=[html.Div(id='container-button-load-internal'),
                                                              dbc.CardColumns(
                                                                  [
                                                                      dbc.Card(
                                                                          [
                                                                              dbc.CardHeader("Built-in: Cats and Dogs",
                                                                                             className='card text-white bg-primary mb-3'),
                                                                              dbc.CardBody(
                                                                                  [
                                                                                      dbc.CardText(
                                                                                          "This dataset is internally available and provides labelled training/validation\nimages for classifying dogs and cats"),
                                                                                      html.Button('Load',
                                                                                                  id='btn-load-internal',
                                                                                                  className='btn btn-primary')
                                                                                  ]
                                                                              ),
                                                                          ]
                                                                      )])])

    return output


def getContentExplore():
    global source_state
    global train_cat_fnames
    global train_dog_fnames
    global train_cats_dir
    global train_dogs_dir
    global validation_cats_dir
    global validation_dogs_dir

    if source_state == 1:
        output = html.Div([dbc.Alert("Error: Dataset is not yet loaded", color="danger"), ])

    elif source_state == 2:
        # Load internal dogs/cats
        # listDataSet()

        # Hacky workaround to show dynamic graphics in Dash
        buf = io.BytesIO()
        getSourceDataExampleImg().savefig(buf, format='png')
        buf.seek(0)
        encoded_image = base64.b64encode(buf.getvalue()).decode()
        output = html.Div([dbc.Card(
            [
                dbc.CardHeader("Dataset Loaded", className='card text-white bg-primary mb-6'),
                dbc.CardBody(
                    [
                        html.Div([html.Pre('total training cat images:' + str(len(os.listdir(train_cats_dir))) +
                                           '\r\ntotal training dog images:' + str(
                            len(os.listdir(train_dogs_dir))) + '\r\ntotal validation cat images:' +
                                           str(len(os.listdir(
                                               validation_cats_dir))) + '\r\ntotal validation dog images:' + str(
                            len(os.listdir(validation_dogs_dir))))])
                    ])]),
            html.Div(style={'height': '25px'}),
            dbc.Card(
                [
                    dbc.CardHeader("Example images", className='card text-white bg-primary mb-6'),
                    dbc.CardBody(
                        [
                            html.Img(src='data:image/png;base64,{}'.format(encoded_image), width='90%')
                        ])
                ])])

    return output


def getModelSummary():
    if source_state == 1:
        output = html.Div([dbc.Alert("Error: Dataset is not yet loaded", color="danger"), ])
    else:

        global model
        global img_input

        model, img_input = getModel()

        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)

        from tensorflow.keras.utils import plot_model
        import base64

        import tempfile
        new_file, filename = tempfile.mkstemp()
        filename = filename + '.png'

        print(filename)

        plot_model(model, to_file=filename, show_shapes=True, show_layer_names=True)

        encoded_image = base64.b64encode(open(filename, 'rb').read()).decode()
        os.remove(filename)
        # print(encoded_image[1:10])

        output = html.Div([dbc.Card(
            [
                dbc.CardHeader("Model Details", className='card text-white bg-primary mb-4'),
                dbc.CardBody(
                    [
                        html.Div([html.Pre(short_model_summary)])
                    ])
            ]), html.Div(style={'height': '25px'}),
            dbc.Card(
                [
                    dbc.CardHeader("Model Visualization", className='card text-white bg-primary mb-4'),
                    dbc.CardBody(
                        [
                            html.Img(src='data:image/png;base64,{}'.format(encoded_image), width='450px')
                        ])
                ])])

    return output


def getContentTraining():
    global model
    global source_state

    if source_state == 1:
        output = html.Div([dbc.Alert("Error: Dataset is not yet loaded", color="danger"), ])
        return (output)
    elif model == None:
        output = html.Div([dbc.Alert("Error: Model not yet created. Please go to 'Model Summary' and check the model",
                                     color="danger"), ])
        return (output)

    global training_log

    # learning_rate = 0.001, optimizer = 'RMSprop',epochs=3

    output = html.Div([
        html.Div(id='container-button-start-train'), html.Div(style={'height': '25px'}),
        dbc.Card(
            [
                dbc.CardHeader("Input parameters", className='card text-white bg-primary mb-4'),
                dbc.CardBody(
                    [
                        # Div table
                        html.Div([
                            # Div table body
                            html.Div([
                                # Div table row
                                html.Div([
                                    # Div table cell
                                    html.Div([html.P('Optimizer: '), dcc.Dropdown(id='train-optimizer',
                                                                                  options=[
                                                                                      {'label': 'RMSprop',
                                                                                       'value': 'RMSprop'},
                                                                                      {'label': 'Adam', 'value': 'Adam'}
                                                                                  ], style={'width': '250px'},
                                                                                  value='RMSprop')],
                                             style={'display': 'table-cell', 'padding': '3px 10px'}),
                                    # Div table cell
                                    html.Div(
                                        [html.P('No. of Epochs: '), html.Div(id='update-epoch-slides-output-container')
                                            , dcc.Slider(id='train-epochs', min=1, max=20, step=1, value=3)],
                                        style={'display': 'table-cell', 'padding': '3px 50px'}),
                                    # Div table cell
                                    html.Div([html.P('Learning rate: '),
                                              dcc.Input(id='train-learningrate', placeholder='Enter a value...',
                                                        type='text', value='0.001')],
                                             style={'display': 'table-cell', 'padding': '3px 10px'}),
                                    # Div table cell
                                    html.Div([html.Button('Train Model', id='btn-start-train',
                                                          className='btn btn-primary')])],
                                    style={'display': 'table-cell', 'padding': '3px 10px'})
                            ], style={'display': 'table-row'})
                        ], style={'display': 'table-row-group'})
                    ], style={'wdith': '75%', 'display': 'table'})
            ]), html.Div(style={'height': '25px'}),
        dbc.Card(
            [
                dbc.CardHeader("Training Model", className='card text-white bg-primary mb-4'),
                dbc.CardBody(
                    [
                        html.Div([html.Div(id='pre_log', children=[html.Pre(training_log)],
                                           style={'height': '500px', 'display': 'block'}),
                                  html.Div(id='interval-container', children=[dcc.Interval(
                                      id='interval-trainingoutput',
                                      interval=1 * 1000,  # in milliseconds
                                      n_intervals=0
                                  )])
                                  ])
                    ], style={'height': '550px', 'overflow-y': 'scroll'})

            ])])

    return output


def getContentResults():
    global model
    global source_state
    global training_state
    global source_data_classes

    if source_state == 1:
        output = html.Div([dbc.Alert("Error: Dataset is not yet loaded", color="danger"), ])
        return (output)
    elif model == None:
        output = html.Div([dbc.Alert("Error: Model not yet created. Please go to 'Model Summary' and check the model",
                                     color="danger"), ])
        return (output)
    elif training_state < 2:
        output = html.Div([dbc.Alert("Error: Model not trained!", color="danger")])
        return (output)

    global history

    # Hacky workaround to show dynamic graphics in Dash
    buf = io.BytesIO()
    getLossPlot(history).savefig(buf, format='png')
    buf.seek(0)
    encoded_image_plot = base64.b64encode(buf.getvalue()).decode()

    # Example prediction
    from tensorflow.keras.models import load_model
    trained_model = load_model("./export/model_trained.h5")
    image_path, prob = predictOnRandomTestImage(trained_model)

    encoded_image_pred = base64.b64encode(open(image_path, 'rb').read()).decode()

    output = html.Div([dbc.Card(
        [
            dbc.CardHeader("Prediction", className='card text-white bg-primary mb-6'),
            dbc.CardBody(
                [
                    html.Div(id='container-pred-result', children=[
                        html.Img(src='data:image/png;base64,{}'.format(encoded_image_pred), height='250px'),
                        html.Br([]),
                        html.P(
                            ['Probability of being a ', html.B(source_data_classes[0]), ' is ', html.B(str(prob[0]))])
                    ]), html.Br([]),
                    html.Button('Try Prediction', id='btn-retry-predict', className='btn btn-primary', n_clicks=None)
                ])  # end container
        ]),
        dbc.Card(
            [
                dbc.CardHeader("Training loss/validation loss", className='card text-white bg-primary mb-6'),
                dbc.CardBody(
                    [
                        html.Img(src='data:image/png;base64,{}'.format(encoded_image_plot), height='550px')
                    ])
            ])])

    return output

# predictOnRandomTestImage()


def flatten(l): return flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]


def getContentDeepLayers():
    import io
    global model
    global source_state
    global training_state
    global source_data_classes

    if source_state == 1:
        output = html.Div([dbc.Alert("Error: Dataset is not yet loaded", color="danger"), ])
        return (output)
    elif model == None:
        output = html.Div([dbc.Alert("Error: Model not yet created. Please go to 'Model Summary' and check the model",
                                     color="danger"), ])
        return (output)
    elif training_state < 2:
        output = html.Div([dbc.Alert("Error: Model not trained!", color="danger")])
        return (output)

    img_collection = []
    plot_collection = plotDeepLayers()

    for img in plot_collection:
        img_collection.append(html.Img(src='data:image/png;base64,{}'.format(img), width='98%'))

    output = html.Div([dbc.Card(
        [
            dbc.CardHeader("Intermediate DNN Representation", className='card text-white bg-primary mb-6'),
            dbc.CardBody(
                [
                    html.Div(id='container-deeplayers', children=flatten(img_collection)), html.Br([])  # ,
                    # html.Button('Try image', id='btn-retry-deeplayers',className='btn btn-primary',n_clicks=None)
                ])
        ])])

    return (output)


#############################################
# Interaction Between Components / Controller
#############################################

@app.callback(Output('card-tabs', 'active_tab'),
              [Input('btn-reset', 'n_clicks')]
              )
def reset_app(click):
    resetAppState()
    return ('tab-welcome')


@app.callback(Output('btn-load-internal', 'n_clicks'),
              [Input('btn-reset', 'n_clicks')])
def update(reset):
    return 0


# @app.callback(Output('navheader','children'),
@app.callback(Output('container-button-reset', 'children'),
              [Input('btn-reset', 'n_clicks')])
def update_header(clicks):
    if (clicks > 0):
        return html.Script('location.reload(true);', type='javascript')


@app.callback(
    Output("card-content", "children"), [Input("card-tabs", "active_tab"), Input("btn-reset", "n_clicks")]
)
def tab_content(active_tab, reset_clicks):
    if active_tab == 'tab-sourcedata':
        return getContentSourceData()
    elif active_tab == 'tab-exploredata':
        return getContentExplore()
    elif active_tab == 'tab-modelsummary':
        return getModelSummary()
    elif active_tab == 'tab-training':
        return getContentTraining()
    elif active_tab == 'tab-results':
        return getContentResults()
    elif active_tab == 'tab-layers':
        return getContentDeepLayers()
    return "This is tab {}".format(active_tab)


@app.callback(Output('container-button-load-internal', 'children'),
              [Input('btn-load-internal', 'n_clicks')]
              )
def update_data(n_clicks):
    global source_state
    global source_data_classes

    if n_clicks != None:
        source_state = 2

        # Load internal dogs/cats
        source_data_classes = ['cats', 'dogs']
        listDataSet()
        createGenerators(labelClasses=source_data_classes)

        output = html.Div([dbc.Alert("Dataset loaded!", color="success"), ])
        return output

    else:
        return True
#        source_state = 1
#    return output


@app.callback(Output('pre_log', 'children'),
              [Input('interval-trainingoutput', 'n_intervals')],
              [State('btn-start-train','n_clicks')]
)
def update_metrics(n,n_clicks):
    if n_clicks != None:
        global training_log
        global training_state
        if training_state == 2:
            output = [dbc.Alert("Training in completed!", color="success"),html.Pre(newString(training_log))]
        else:
            output = [dbc.Alert("Training in progress - Please wait " + str('.' * (n % 10)), color="primary"),html.Pre(newString(training_log))]
        return output


@app.callback(Output('update-epoch-slides-output-container', 'children'),
              [Input('train-epochs', 'value')])
def display_value(value):
    return (value)


@app.callback(Output('container-button-start-train', 'children'),
    [Input('btn-start-train','n_clicks')],
              [State('train-optimizer','value'),State('train-epochs','value'),State('train-learningrate','value')]
)
def start_train(click,optimizer_value,no_epochs,learning_rate):
    if click != None:
        StartTraining(learning_rate=float(learning_rate),optimizer=str(optimizer_value),epochs=int(no_epochs))
        return(dbc.Alert("Model trained - Go to next tab to inspect the results!", color="success"))
    elif click == 1:
        return(dbc.Alert("Training in progress - Please wait", color="primary"))


@app.callback(Output('container-pred-result', 'children'),
              [Input('btn-retry-predict', 'n_clicks')]
              )
def retry_predict(click):
    if click != None:
        if click > 0:
            # Example prediction
            from tensorflow.keras.models import load_model
            trained_model = load_model("./export/model_trained.h5")
            image_path, prob = predictOnRandomTestImage(trained_model)

            encoded_image_pred = base64.b64encode(open(image_path, 'rb').read()).decode()

            output = [
                # html.Button('Try another prediction', id='btn-retry-predict',className='btn btn-primary',n_clicks=0),html.Br([]),
                html.Img(src='data:image/png;base64,{}'.format(encoded_image_pred), height='250px'), html.Br([]),
                html.P(['Probability of being a ', html.B(source_data_classes[0]), ' is ', html.B(str(prob[0]))]),
                html.Br([])
            ]

            return output


@app.callback(Output('container-button-load', 'children'),
              [Input('btn-load-model', 'n_clicks')]
              )
def load_model(click):
    if click != None:
        global training_state
        global model
        global history
        import pickle

        from tensorflow.keras import Model
        from tensorflow.keras.models import load_model

        model = load_model("./export/model_best.h5")

        pickle_in = open("./export/model_best_history.pickle", "rb")
        history = pickle.load(pickle_in)
        pickle_in.close()

        training_state = 2

        # Make sure we have the intermediate file saved, for usage in other parts of the code
        model.save('export/model_trained.h5')
        pickle_out = open("./export/model_trained_history.pickle", "wb")
        pickle.dump(history, pickle_out)
        pickle_out.close()

    return ['']


@app.server.route('/images/<path:path>')
def send_image(path):
    return flask.send_from_directory('app_images', path)


# start Flask server
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8802,threaded=True)

