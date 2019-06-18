# NN-for-PLINK

**Specify hidden and output layers**
```
weights={
    'W_hidden_1': tf.get_variable("W1", shape=#your shape,initializer=tf.contrib.layers.xavier_initializer()),
    'W_hidden_2': tf.get_variable("W2", shape=#your shape,initializer=tf.contrib.layers.xavier_initializer()),
    'W_out': tf.get_variable("W3", shape=#your shape,initializer=tf.contrib.layers.xavier_initializer())
}

biases={

'b1' : tf.Variable(tf.random_normal([#your dim])),

'b2' : tf.Variable(tf.random_normal([#your dim])),

'b3' : tf.Variable(tf.random_normal([#your dim]))

}
```


**Calculate outputs from each layer**
```
def run_NN(input_x, W, B):
    # Hidden layer
    hidden_1 = tf.nn.relu(tf.add(tf.matmul(input_x, W['W_hidden_1'] ), B['b1']))
    hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1 , W['W_hidden_2']), B['b2']))
# Output layer (must be transposed)
    out_pre = tf.transpose(tf.add(tf.matmul(hidden_2, W['W_out']), B['b3']))
    out=tf.tanh(out_pre)
    return out
```


**Calcuate the output, accuarcy, and loss**
```
out=run_NN(X,weights, biases)

#cost 
cost=tf.reduce_mean(tf.squared_difference(out, Y))

# Optimizer
opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

prediction=out
equals=tf.equal(prediction, Y)
accuracy = tf.reduce_mean(tf.cast(equals, tf.float32))
```

**Run the model using batch gradient descent**
```
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
 ```
