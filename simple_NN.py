

export LD_LIBRARY_PATH="/home/cmb-07/sn1/mingyous/anaconda2/lib64:$LD_LIBRARY_PATH"
export PATH="/home/cmb-07/sn1/mingyous/anaconda2/bin:$PATH"
source activate deeplift

cd  /home/cmb-07/sn1/mingyous/oilpalm/newdata/


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import data
data = pd.read_csv('SOL_OILPALM.Eguineensis9.1.VF.SNP.DP8.20chr.chr.id.OG.dp10.parents.nomissing.noparents.beagle.snpEff_genic.pruned.noCor.raw',delimiter=' ',header=0)
#data.head(0)
pheno = pd.read_csv('OGpheno1sampleMatch.txt',delimiter='\t',header=0)

merged=pd.merge(pheno,data,on='IID')

temp2=data.columns.values
chr3=np.flatnonzero(np.array([i.startswith('3:') for i in temp2])!=False)

data=merged.iloc[:,chr3]
pheno=merged.iloc[:,1]

data=data.values #numpy format array
pheno=pheno.values


#shuffle data
shuffle_indices = np.random.permutation(np.arange(data.shape[0]))
data=data[shuffled_indices,:]


#scale to -1~1
meaned=(pheno-np.mean(pheno))
pheno=meaned/np.max(np.absolute(meaned))

# Dimensions of dataset
n = data.shape[0]
p = data.shape[1]


# Import TensorFlow
import tensorflow as tf

# Model architecture parameters
input_dim = p
n_neurons_1 = input_dim
n_neurons_2 = input_dim
n_neurons_3 = input_dim

n_target = 1

# Placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, input_dim])
Y = tf.placeholder(dtype=tf.float32, shape=[None])

#xavier_initializer
#https://adventuresinmachinelearning.com/weight-initialization-tutorial-tensorflow/
#regularization
#http://laid.delanover.com/difference-between-l1-and-l2-regularization-implementation-and-visualization-in-tensorflow/

# Layer 1: Variables for hidden weights and biases
W_hidden_1=tf.get_variable("W1", shape=[input_dim, n_neurons_1],
                         initializer=tf.contrib.layers.xavier_initializer())
#This allows us to pass in a custom initializer for our weights
b1 = tf.Variable(tf.random_normal([n_neurons_1]))
# Layer 2: Variables for hidden weights and biases
W_hidden_2=tf.get_variable("W2", shape=[n_neurons_1, n_neurons_2],
                         initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([n_neurons_2]))

# Output layer: Variables for output weights and biases
W_out = tf.get_variable("W3", shape=[n_neurons_2, n_target],
                         initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([n_target]))

# Hidden layer
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1 ), b1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1 , W_hidden_2), b2))

# Output layer (must be transposed)
out_pre = tf.transpose(tf.add(tf.matmul(hidden_2, W_out), b3))
out=tf.tanh(out_pre)

#L2
cost=tf.reduce_mean(tf.squared_difference(out, Y))

# Optimizer
opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

prediction=out
equals=tf.equal(prediction, Y)
accuracy = tf.reduce_mean(tf.cast(equals, tf.float32))

# Make Session
net = tf.Session()
# Run initializer
net.run(tf.global_variables_initializer())


# Number of epochs and batch size
epochs = 200
batch_size = 77



#cv
cv_fold=8
cv_elCount=n//cv_fold
#accuracy array
accuracy_all=np.array([])
#for cv in np.arange(cv_fold):
cv=1
# Training and test data
test_start = cv * cv_elCount
test_end = cv * cv_elCount + cv_elCount
total_set=np.arange(n)
train_set = np.setdiff1d(total_set,np.arange(test_start,test_end))
#
data_train_x = data[train_set, :]
data_test_x = data[np.arange(test_start, test_end), :]
data_train_y = pheno[train_set]
data_test_y = pheno[np.arange(test_start, test_end)]
# Build X and y
X_train = data_train_x
y_train = data_train_y
X_test = data_test_x
y_test = data_test_y
# Setup interactive plot
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test)
line2, = ax1.plot(y_test*0.5)
plt.show()
avg_cost = 0
for e in range(epochs):
    # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size,:]
        batch_y = y_train[start:start + batch_size]
        # Run optimizer with batch
        _, c = net.run([opt,cost], feed_dict={X: batch_x, Y: batch_y})
        print(c)
        # Show progress
        if np.mod(i, 10) == 0:
            # Prediction
            pred = net.run(prediction, feed_dict={X: X_test})
            line2.set_ydata(pred)
            plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
            file_name = 'epoch_' + str(e) + '_batch_' + str(i) + '.png'
            plt.savefig(file_name)
            plt.pause(0.01)
        avg_cost += c / (batch_size*epochs)
        print("Epoch:", (e + 1),  ' Batch ' + str(i), "cost =", "{:.10f}".format(avg_cost))

pred = net.run(prediction, feed_dict={X: X_test})
# Print final MSE after Training
mse_final = net.run(accuracy, feed_dict={X: X_test, Y: y_test})
print(mse_final)
accuracy_all=np.append(accuracy_all,mse_final)
