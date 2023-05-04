# import modules

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# loading data

data = tf.keras.datasets.cifar10

(train_ , train_lbl) , (test_ , test_lbl) = data.load_data()

# look at the data

print(f'the shape of train data is {train_.shape}')
print(f'the shape of test data is {test_.shape}')

print(f'the shape of train labels is {train_lbl.shape}')
print(f'the shape of test labels data is {test_lbl.shape}')

# convert numbers into labels and create a num classes

class_names = ['airplan' , 'automobil' , 'bird' , 'cat' , 'deer' , 'dog' , 'frog' , 'horse' , 'ship' , 'truck']
num_values = list(set(train_lbl.reshape((50000)).tolist()))

num_classes = dict(zip(num_values , class_names))

num_classes

# look at the images 

import random

plt.figure(figsize = (4,4))

img_indx = random.randint(0,50000)
plt.imshow(train_[img_indx])
plt.colorbar()
plt.title(num_classes[train_lbl[img_indx].tolist()[0]])

plt.show()

# convert image pixels between 0 and 1

train_ , test_ = train_ / 255.0 , test_ / 255.0


model = tf.keras.Sequential([
    
    # Convolutional layers
    tf.keras.layers.Conv2D(32 , (3,3) , activation = 'relu' , input_shape = (32,32 , 3)) ,
    tf.keras.layers.MaxPool2D((2,2)) ,
    tf.keras.layers.Conv2D(64 , (3,3) , activation = 'relu') ,
    tf.keras.layers.MaxPool2D((2,2)) ,
    tf.keras.layers.Conv2D(64 , (3,3) , activation = 'relu') ,

    # classifier layers

    tf.keras.layers.Flatten() , 
    tf.keras.layers.Dense(64 , activation = 'relu') ,
    tf.keras.layers.Dropout(0.5) , 
    tf.keras.layers.Dense(10) , 

])

model.summary()

# compile the model

op = tf.keras.optimizers.Adam()
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer = op , loss = loss , metrics = ['accuracy'])

# trsin the model

epoch = 20
his = model.fit(train_ , train_lbl , epochs=epoch , validation_data=(test_ , test_lbl))

# check how our model learn

model.evaluate(test_ , test_lbl , verbose=0)

# plot the information of model

plt.figure(figsize = (5 , 5))

loss_spot = his.history['loss']
acc_spot = his.history['accuracy']
val_loss_spot = his.history['val_loss']
val_acc_spot = his.history['val_accuracy']

epoch_range = range(epoch)

fig , axis = plt.subplots(1,2)

axis[0].plot(epoch_range , acc_spot , label = 'accuracy')
axis[0].plot(epoch_range , val_acc_spot , label = 'vallidation accuracy')
plt.legend()


axis[1].plot(epoch_range , loss_spot , label = 'loss')
axis[1].plot(epoch_range , val_loss_spot , label = 'vallidation loss')

axis[0].legend()
axis[1].legend()
plt.show()

# makes predictions list

prediction = model.predict(test_)


# look how our model predict the results

plt.figure(figsize = (10 ,10))

for i in range(0 , 25) : 

    plt.subplot(5,5,i+1)
    plt.imshow(test_[i])
    plt.xticks([])
    plt.yticks([])

    pre = np.argmax(prediction[i])

    color = 'blue' if test_lbl[i] == pre else 'red'
    plt.title(num_classes[pre] , color = color) 


plt.show()

# you can see how one convolutional layer work with activation relu 

plt.figure(figsize = (4 ,4))
fill = 32
img = test_[60]
con = tf.keras.layers.Conv2D(fill,(3,3) , activation = 'relu' , input_shape = (32,32,3))
m1 = np.squeeze(con(img[np.newaxis]))
con = tf.keras.layers.MaxPool2D((2,2))
m1 = con(m1[np.newaxis])

fig , ax = plt.subplots(1,2)
ax[1].imshow(m1[0][:,:,fill-1])
ax[0].imshow(img)

plt.show()

# you can see how one convolutional layer work with activation sigmoid 

plt.figure(figsize = (4 ,4))
fill = 32
img = test_[60]
con = tf.keras.layers.Conv2D(fill,(3,3) , activation = 'sigmoid' , input_shape = (32,32,3))
m1 = np.squeeze(con(img[np.newaxis]))
con = tf.keras.layers.MaxPool2D((2,2))
m1 = con(m1[np.newaxis])

fig , ax = plt.subplots(1,2)
ax[1].imshow(m1[0][:,:,fill-1])
ax[0].imshow(img)



plt.show()

# you can see how one convolutional layer work with activation softmax 

plt.figure(figsize = (4 ,4))
fill = 32
img = test_[60]
con = tf.keras.layers.Conv2D(fill,(3,3) , activation = 'softmax' , input_shape = (32,32,3))
m1 = np.squeeze(con(img[np.newaxis]))
con = tf.keras.layers.MaxPool2D((2,2))
m1 = con(m1[np.newaxis])

fig , ax = plt.subplots(1,2)
ax[1].imshow(m1[0][:,:,fill-1])
ax[0].imshow(img)



plt.show()

# you can see how one convolutional layer work with activation tanh 

plt.figure(figsize = (4 ,4))
fill = 32
img = test_[60]
con = tf.keras.layers.Conv2D(fill,(3,3) , activation = 'tanh' , input_shape = (32,32,3))
m1 = np.squeeze(con(img[np.newaxis]))
con = tf.keras.layers.MaxPool2D((2,2))
m1 = con(m1[np.newaxis])

fig , ax = plt.subplots(1,2)
ax[1].imshow(m1[0][:,:,fill-1])
ax[0].imshow(img)



plt.show()

