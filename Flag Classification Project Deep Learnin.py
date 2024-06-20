import numpy as np 
import zipfile 
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array 


zip_fil = 'myanmar-enthic-flags.zip'
with zipfile.ZipFile(zip_fil, 'r') as fil:
    fil.extractall()

train_dir = 'ds_split/train'
val_dir = 'ds_split/val'

#Data preprocessing 

train_datagen = ImageDataGenerator(
    rescale = 1./255.0,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    zoom_range= 0.2,
    shear_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest'
)

val_datagen = ImageDataGenerator(
    rescale = 1./255.0,
    width_shift_range =0.2,
    height_shift_range = 0.2,
    zoom_range = 0.2,
    shear_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest'
)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size = (228,228),
    batch_size = 32,
    class_mode = 'categorical'
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size = (228, 228),
    batch_size = 32,
    class_mode = 'categorical'
)

num_classes = 8

#Modeling Deep Learning Architecture 
model = tf.keras.Sequential([
    
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (228, 228, 3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(128, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(num_classes, activation = 'softmax')
])

#Compiling the model 
model.compile(loss = 'categorical_crossentropy',
              optimizer ='adam',
              metrics = ['accuracy'])

#Fit the model 
model.fit(train_gen, epochs = 20, validation_data = val_gen)

#Apply Function for prediction 
def predict_flag(model, img_path, img_size = (228, 228)):
    img = load_img(img_path, target_size = img_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis = 0)
    img_array  = img_array/255.0
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis = 1)[0]
    label_map = {0 : 'Kachin', 1 : 'Kayah',
                 2 : 'Karen', 3 : 'Chin',
                 4 : 'Mon', 5 : 'Burma',
                 6 : 'Rakhine', 7 : 'Shan'}

    predicted_country = label_map[predicted_class]
    return predicted_country, prediction[0]


img_path ='ds_split/test/flag_6_0026.jpg'
prediction, probabilities = predict_flag(model, img_path)
print(f"Predictions: {prediction}")
print(f"Probabilities: {probabilities}")