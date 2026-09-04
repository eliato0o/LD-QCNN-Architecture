!pip install tensorflow==2.8.0
!pip install tensorflow-quantum==0.7.2

# Update package resources to account for version changes.
import importlib, pkg_resources
importlib.reload(pkg_resources)







import tensorflow as tf
import numpy as np
import tensorflow_quantum as tfq

import cirq
import sympy
import seaborn as sns
import collections

# visualization tools
%matplotlib inline
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit





##################################################### DATASET CONFIGURATION ############################################################



(x_training, y_training), (x_testing, y_testing) = tf.keras.datasets.mnist.load_data()

# Rescale the images from [0,255] to the [0.0,1.0] range.
x_training, x_testing = x_training[..., np.newaxis]/255.0, x_testing[..., np.newaxis]/255.0

print("Number of original training examples:", len(x_training))
print("Number of original testing examples:", len(x_testing))




def filter_07(r, n):
    keep = (n == 0) | (n == 7)
    r, n = r[keep], n[keep]
    n = n == 0
    return r,n


x_training_07, y_training_07 = filter_07(x_training, y_training)
x_testing_07, y_testing_07 = filter_07(x_testing, y_testing)

print("Number of filtered training examples:", len(x_training_07))
print("Number of filtered testing examples:", len(x_testing_07))

print(y_training_07[0])

plt.imshow(x_training_07[0, :, :, 0])
plt.colorbar()



x_train_sized = tf.image.resize(x_training_07, (4,4)).numpy()
x_test_sized = tf.image.resize(x_testing_07, (4,4)).numpy()

print(y_training_07[0])
plt.imshow(x_train_sized[0,:,:,0], vmin=0, vmax=1)
plt.colorbar() 





def remove_contradicting(xs, ys):
    mapping = collections.defaultdict(set)
    for x,y in zip(xs,ys):
       mapping[tuple(x.flatten())].add(y)
    
    new_x = []
    new_y = []
    for x,y in zip(xs, ys):
      labels = mapping[tuple(x.flatten())]
      if len(labels) == 1:
          new_x.append(x)
          new_y.append(list(labels)[0])
      else:
          pass
    
    num_0 = sum(1 for value in mapping.values() if True in value)
    num_7 = sum(1 for value in mapping.values() if False in value)
    num_both = sum(1 for value in mapping.values() if len(value) == 2)

    print("Number of unique images in the dataset:", len(mapping.values()))
    print("Number of zeroes: ", num_0)
    print("Number of sevens: ", num_7)
    print("Total Number of contradictory images: ", num_both)
    print()
    print("Initial number of examples in the datset: ", len(xs))
    print("Remaining non-contradictory examples for use: ", len(new_x))
    
    return np.array(new_x), np.array(new_y)

x_training_nocont, y_training_nocontt = remove_contradicting(x_train_sized, y_training_07)

Threshold = 0.5
x_train_binary = np.array(x_training_nocont > Threshold, dtype=np.float32)
x_test_binary = np.array(x_test_sized > Threshold, dtype=np.float32)

len(x_train_binary)

x_train_binary_nocont,y_training_nocont = remove_contradicting(x_train_binary, y_training_nocontt)




















######################################################## ENCODING DU DATASET POUR LE CIRCUIT ####################################################################



def convert_to_circuit(image):
    values = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(4, 4)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        if value:
            circuit.append(cirq.X(qubits[i]))
    return circuit


x_train_circuit = [convert_to_circuit(x) for x in x_train_binary_nocont]
x_test_circuit = [convert_to_circuit(x) for x in x_test_binary]

len(x_train_circuit)


binary_image = x_train_binary[0,:,:,0]
index = np.array(np.where(binary_image)).T
index





x_train_circuit_tensor = tfq.convert_to_tensor(x_train_circuit)
x_test_circuit_tensor = tfq.convert_to_tensor(x_test_circuit)

x_train_binary[0].shape




# Simple représentation du circuit ?

class CircuitLayerBuilder():
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout
    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout)**symbol)



demo_circuit = CircuitLayerBuilder(data_qubits = cirq.GridQubit.rect(4,1),
                                   readout=cirq.GridQubit(-1,-1))
circ = cirq.Circuit()
demo_circuit.add_layer(circ, gate = cirq.XX, prefix='xx')
SVGCircuit(circ) 




















############################################################## CONSTRUCTION DU MODELE ##############################################################################















def create_quantum_model():
    data_qubits = cirq.GridQubit.rect(4, 4)  # a 4x4 grid.
    readout = cirq.GridQubit(-1, -1)         # a single qubit at [-1,-1]
    circuit = cirq.Circuit()
    
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))
    
    builder = CircuitLayerBuilder(
        data_qubits = data_qubits,
        readout=readout)

    builder.add_layer(circuit, cirq.XX, "xx1")
    builder.add_layer(circuit, cirq.ZZ, "zz1")

    circuit.append(cirq.H(readout))
    circuit.append(cirq.measure(readout))

    return circuit, cirq.Z(readout)



model_circuit, model_readout = create_quantum_model()
SVGCircuit(model_circuit)  # ??????????



















model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(), dtype=tf.string),
    tfq.layers.PQC(model_circuit, model_readout),    
])

y_training_hinge = 2.0*y_training_nocont-1.0
y_testing_hinge = 2.0*y_testing_07-1.0

y_testing_07


def measure_hinge_accuracy(y_true, y_predicted):
    y_true = tf.squeeze(y_true) > 0.0
    y_predicted = tf.squeeze(y_predicted) > 0.0
    result = tf.cast(y_true == y_predicted, tf.float32)

    return tf.reduce_mean(result)



model.compile(
    loss=tf.keras.losses.mean_squared_error,
    optimizer=tf.keras.optimizers.Adam(lr=0.01),
    metrics=[measure_hinge_accuracy])


print(model.summary())


































###################################################### TRAINING ##############################################################








EPOCHS = 10
BATCH_SIZE = 32

NUM_EXAMPLES = len(x_train_circuit_tensor)


len(x_test_circuit_tensor)

x_train_circuit_tensor_subset = x_train_circuit_tensor[:NUM_EXAMPLES]
y_train_hinge_subset = y_training_hinge[:NUM_EXAMPLES]

len(x_train_circuit_tensor_subset)



class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
 
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))




import time
history = LossHistory()
begin_time = time.time()
qnn_history = model.fit(
      x_train_circuit_tensor_subset, y_train_hinge_subset,
      batch_size=32,
      epochs=EPOCHS,
      verbose=1,
      validation_data=(x_test_circuit_tensor, y_testing_hinge),
      callbacks=[history])
end_time = time.time()
print('the cost time is '+str(end_time-begin_time))
QNN_results= model.evaluate(x_test_circuit_tensor, y_testing_07)









loss = qnn_history.history['loss']
hinge_acc = qnn_history.history['measure_hinge_accuracy']

val_hinge_acc = qnn_history.history['val_measure_hinge_accuracy']
val_loss = qnn_history.history['val_loss']







epochs_arr = range(1, len(hinge_acc) + 1)
plt.plot(epochs_arr, hinge_acc, 'r', label='Training acc')
plt.plot(epochs_arr, val_hinge_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()






plt.plot(epochs_arr, loss, 'r', label='Training loss')
plt.plot(epochs_arr, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


QNN_results


len(x_train_binary)
len(x_test_binary)
len( y_training_07)
len(y_training_nocont)



model.evaluate(x_test_circuit_tensor, y_testing_07)








############################################# CLASSICAL CNN ##########################################################





def CNN_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, [3, 3], activation='relu', input_shape=(28,28,1)))
    model.add(tf.keras.layers.Conv2D(64, [3, 3], activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1))
    return model


model = CNN_model()
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

model.summary()


history = LossHistory()
begin_time = time.time()
model.fit(x_training_07,
          y_training_07,
          batch_size=128,
          epochs=5,
          verbose=1,
          validation_data=(x_testing_07, y_testing_07),
          callbacks=[history])
end_time = time.time()
print('the cost time is '+str(end_time-begin_time))
CNN_results = model.evaluate(x_testing_07, y_testing_07)

len(history.losses)







############################################# CLASSICAL ANN ##########################################################








def ANN_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(4,4,1)))
    model.add(tf.keras.layers.Dense(2, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    return model


model = ANN_model()
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

model.summary()




model.fit(x_train_binary,
          y_training_nocontt,
          batch_size=128,
          epochs=20,
          verbose=2,
          validation_data=(x_test_binary, y_testing_07))

ANN_results = model.evaluate(x_test_binary, y_testing_07)


ANN_results[1]















QNN_accuracy = QNN_results[1]
CNN_accuracy = CNN_results[1]
ANN_accuracy = ANN_results[1]

plt.bar(["QNN", "CNN", "ANN"],[QNN_accuracy, CNN_accuracy, ANN_accuracy])


























############################################# QCNN ##########################################################



def cluster_state_circuit(bits):
    circuit = cirq.Circuit()
    circuit.append(cirq.H.on_each(bits))
    for this_bit, next_bit in zip(bits, bits[1:] + [bits[0]]):
        circuit.append(cirq.CZ(this_bit, next_bit))
    return circuit


SVGCircuit(cluster_state_circuit(cirq.GridQubit.rect(1, 8)[0::2]))




def one_qubit_unitary(bit, symbols):
    return cirq.Circuit(
        cirq.X(bit)**symbols[0],
        cirq.Y(bit)**symbols[1],
        cirq.Z(bit)**symbols[2])


def two_qubit_unitary(bits, symbols):
    circuit = cirq.Circuit()
    circuit += one_qubit_unitary(bits[0], symbols[0:3])
    circuit += one_qubit_unitary(bits[1], symbols[3:6])
    circuit += [cirq.ZZ(*bits)**symbols[7]]
    circuit += [cirq.YY(*bits)**symbols[8]]
    circuit += [cirq.XX(*bits)**symbols[9]]
    circuit += one_qubit_unitary(bits[0], symbols[9:12])
    circuit += one_qubit_unitary(bits[1], symbols[12:])
    return circuit


def two_qubit_pool(source_qubit, sink_qubit, symbols):
    pool_circuit = cirq.Circuit()
    sink_basis_selector = one_qubit_unitary(sink_qubit, symbols[0:3])
    source_basis_selector = one_qubit_unitary(source_qubit, symbols[3:6])
    pool_circuit.append(sink_basis_selector)
    pool_circuit.append(source_basis_selector)
    pool_circuit.append(cirq.CNOT(control=source_qubit, target=sink_qubit))
    pool_circuit.append(sink_basis_selector**-1)
    return pool_circuit


def quantum_conv_circuit(bits, symbols):
    circuit = cirq.Circuit()
    for first, second in zip(bits[0::2], bits[1::2]):
        circuit += two_qubit_unitary([first, second], symbols)
    for first, second in zip(bits[1::2], bits[2::2] + [bits[0]]):
        circuit += two_qubit_unitary([first, second], symbols)
    return circuit










qubits=cirq.GridQubit.rect(1, 8)[0::2][0]
symbols = sympy.symbols('qconv0:63')
xxx=two_qubit_pool(cirq.GridQubit.rect(1, 8)[0::2][0],cirq.GridQubit.rect(1, 8)[0::2][1], symbols[0:12])
SVGCircuit(xxx)






def quantum_pool_circuit(source_bits, sink_bits, symbols):
    circuit = cirq.Circuit()
    for source, sink in zip(source_bits, sink_bits):
        circuit += two_qubit_pool(source, sink, symbols)
    return circuit





qubits=cirq.GridQubit.rect(1, 8)[0::2][0]
symbols = sympy.symbols('qconv0:63')
xxx=quantum_pool_circuit(cirq.GridQubit.rect(1, 8)[:4],cirq.GridQubit.rect(1, 8)[4:], symbols[0:12])
SVGCircuit(xxx)


qubits=cirq.GridQubit.rect(1, 8)
symbols = sympy.symbols('qconv0:63')
xxx=quantum_conv_circuit(qubits, symbols[0:15])

SVGCircuit(xxx)









def quantum_conv_circuit(bits, symbols):
    circuit = cirq.Circuit()
    for first, second in zip(bits[0::2], bits[1::2]):
        circuit += two_qubit_unitary([first, second], symbols)
    for first, second in zip(bits[1::2], bits[2::2] + [bits[0]]):
        circuit += two_qubit_unitary([first, second], symbols)
    return circuit



cirq.GridQubit.rect(1, 8)[0::2][0]




def quantum_pool_circuit(source_bits, sink_bits, symbols):
    circuit = cirq.Circuit()
    for source, sink in zip(source_bits, sink_bits):
        circuit += two_qubit_pool(source, sink, symbols)
    return circuit









def create_model_circuit(qubits):
    model_circuit = cirq.Circuit()
    symbols = sympy.symbols('qconv0:21')
    model_circuit += quantum_conv_circuit(qubits, symbols[0:15])
    model_circuit += quantum_pool_circuit(qubits[:4], qubits[4:],
                                          symbols[15:21])
    return model_circuit


cluster_state_bits = cirq.GridQubit.rect(1, 8)
readout_operators = cirq.Z(cluster_state_bits[-1])

excitation_input = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
cluster_state = tfq.layers.AddCircuit()(
    excitation_input, prepend=cluster_state_circuit(cluster_state_bits))

quantum_model = tfq.layers.PQC(create_model_circuit(cluster_state_bits),
                               readout_operators)(cluster_state)

qcnn_model = tf.keras.Model(inputs=[excitation_input], outputs=[quantum_model])

tf.keras.utils.plot_model(qcnn_model,
                          show_shapes=True,
                          show_layer_names=False,
                          dpi=70)












qcnn_model.summary()









def generate_data(qubits):
    n_rounds = 20  # Produces n_rounds * n_qubits datapoints.
    excitations = []
    labels = []
    for n in range(n_rounds):
        for bit in qubits:
            rng = np.random.uniform(-np.pi, np.pi)
            excitations.append(cirq.Circuit(cirq.rx(rng)(bit)))
            labels.append(1 if (-np.pi / 2) <= rng <= (np.pi / 2) else -1)

    split_ind = int(len(excitations) * 0.7)
    train_excitations = excitations[:split_ind]
    test_excitations = excitations[split_ind:]

    train_labels = labels[:split_ind]
    test_labels = labels[split_ind:]

    return tfq.convert_to_tensor(train_excitations), np.array(train_labels), \
        tfq.convert_to_tensor(test_excitations), np.array(test_labels)




train_excitations, train_labels, test_excitations, test_labels = generate_data(cluster_state_bits)
len(x_train_circuit_tensor)
len(y_train_hinge_subset)









qcnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                     loss=tf.losses.mse,
                     metrics=[measure_hinge_accuracy])
begin_time = time.time()
qcnn_history = qcnn_model.fit(x= x_train_circuit_tensor_subset,
                                  y=y_train_hinge_subset,
                                  batch_size=48,
                                  epochs=20,
                                  verbose=1,
                                  validation_data=(x_test_circuit_tensor,
                                                   y_testing_07))
end_time = time.time()
print('the cost time is '+str(end_time-begin_time))







loss = qnn_history.history['loss']
hinge_acc = qnn_history.history['measure_hinge_accuracy']

val_hinge_acc = qnn_history.history['val_measure_hinge_accuracy']
val_loss = qnn_history.history['val_loss']









epochs = range(1, len(hinge_acc) + 1)

plt.plot(epochs, hinge_acc, 'g', label='Training acc')
plt.plot(epochs, val_hinge_acc, 'black', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()






plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'black', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


qcnn_model_results = qcnn_model.evaluate(x_test_circuit_tensor, y_testing_07)
qcnn_model_results


QNN_accuracy = QNN_results[1]
CNN_accuracy = CNN_results[1]
QCNN_accuracy= qcnn_model_results[1]
plt.bar(["QNN", "CNN",  "QCNN"],[QNN_accuracy, CNN_accuracy,QCNN_accuracy])