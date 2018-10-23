from face_recognition import FaceRecognition

if __name__ == '__main__':
    z = FaceRecognition()
    z.load_model_from_file('/Users/francesco/Downloads/Model/vgg16_simpson.json', '/Users/francesco/Downloads/Model/vgg16_simpson_weights.h5')