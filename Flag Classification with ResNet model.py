import zipfile 
import tensorflow as tf 
import numpy as np 
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
from tensorflow.keras.applications.resnet50 import ResNet50 

zip_fil =  'myanmar-enthic-flags.zip'
with zipfile.ZipFile(zip_fil, 'r') as fil:
    fil.extractall() 

#Data preprocessing 
train_dir = 'ds_split/train'
val_dir = 'ds_split/val'

train_gen = ImageDataGenerator(
    rescale = 1./255.0,
    height_shift_range = 0.2,
    width_shift_range = 0.2,
    zoom_range = 0.5,
    shear_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest'
)

val_gen = ImageDataGenerator(
    rescale = 1./255.0,
    height_shift_range = 0.2,
    width_shift_range = 0.2,
    zoom_range = 0.5,
    shear_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest'
)


train_datagen = train_gen.flow_from_directory(
    train_dir,
    target_size= (228, 228),
    batch_size = 32,
    class_mode = 'categorical'
)

val_datagen = val_gen.flow_from_directory(
    val_dir,
    target_size = (228, 228), 
    batch_size = 32,
    class_mode ='categorical'
)

num_classes = 8


#Building and Developing Model Architecture Deep Learning with ResNet50 
basic_model = ResNet50(
    weights = 'imagenet',
    input_shape = (228, 228, 3),
    include_top = False

)

basic_model.trainable = False 
model = tf.keras.Sequential([
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1024, activation = 'relu'),
    tf.keras.layers.Dense(1024, activation = 'relu'),
    tf.keras.layers.Dense(num_classes, activation = 'softmax')
])


#Compiling the model 
model.compile(loss ='categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])


#Fit the model 
model.fit(train_datagen,
          epochs = 10,
          validation_data = val_datagen)
        
    

#Apply function for predict image 
def predict_img(model, img_path, img_size = (228, 228)):
    img = load_img(img_path, target_size = img_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis = 0)
    img_array = img_array /255.0
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis = 1)[0]
    label_map = {
        0 : 'Kachin',
        1 : 'Kayah',
        2 : 'Karen',
        3 : 'Chin',
        4 : 'Mon',
        5 : 'Burma',
        6 : 'Rakhine',
        7 : 'Shan'
    }

    predicted_country = label_map[predicted_class]
    return prediction, predicted_country 


img_path = 'ds_split/test/flags_7_0028.jpg'
prediction, probabilities = predict_img(model, img_path)
print(f'Prediction : {prediction}')
print(f'Probabilities : {probabilities}')


