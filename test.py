from face_recognition import FaceRecognition

train_folder = r'dataset/data/train'
valid_folder = r'dataset/data/validate'

test = FaceRecognition(epochs=10, batch_size=32, image_width=224, image_height=224)
test.create_img_generator()
test.set_train_generator(train_dir=train_folder)
test.set_valid_generator(valid_dir=valid_folder)

# prepare the model
test.set_face_recognition_model(pretrained_model='xception', weights='imagenet')
name = "xception_test_{:d}"
# train fit
test.train_and_fit_model(name)
del test

quit(0)
