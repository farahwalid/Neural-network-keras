# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import keras
from keras.models import Sequential
from keras.layers import Dense
df = pd.read_csv('cardio.csv', sep=';' )
target_column = ['cardio']
pid = ['id']
predictors = list(set(list(df.columns))-set(target_column)-set(pid))
df[predictors] = df[predictors]/df[predictors].max()
df.describe()
X = df[predictors].values
y = df[target_column].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from keras.optimizers import Adam
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=11, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(4, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform',activation='sigmoid'))
    adam = Adam(lr=0.001)
    model.compile(loss="binary_crossentropy", optimizer=adam,metrics=['accuracy'])
    return model
model = create_model()
print(model.summary())
history=model.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=300, batch_size=80,verbose=2)


#plotting
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()
# evaluate the model
_, train_acc = model.evaluate(X_train, y_train, verbose=0)
_, test_acc = model.evaluate(X_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))


############# risk prediction ############
predictions = model.predict_classes(X_test)
# summarize the first 5 cases
for i in range(5):
	print('%s => %d (expected %d)' % (X_test[i].tolist(), predictions[i], y_test[i]))
