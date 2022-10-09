import play_area
from play_area import *
import cnn
from cnn import *
import pickle

def prepare_img(file):
    img = cv.imread(file)
    resized_img = cv.resize(img, (28,28))
    resized_gray = np.expand_dims(cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY), axis=2)
    predict_img = np.expand_dims(resized_gray, axis=0)
    return predict_img

def predict(model, img):
    with open('label_dict.pkl', 'rb') as f:
        label_dict = pickle.load(f)
    predictions = model.predict(img)
    score = tf.nn.softmax(predictions[0]).numpy()
    top_3 = score.argsort()[-3:][::-1]
    print(
        "This image is likely {} with a {:.2f} percent confidence. Next 2 guesses are {} with {:.2f} percent confidence and {} with {:.2f} percent confidence."
        .format(label_dict[top_3[0]], 100 * score[top_3[0]], label_dict[top_3[1]], 100 * score[top_3[1]], label_dict[top_3[2]], 100 * score[top_3[2]])
    )

def run_prediction():
    play_area.find_current_play_area()
    img = prepare_img('drawing.png')
    predict(model,img)

if __name__ == '__main__':
    keyboard.add_hotkey('ctrl+alt+p', run_prediction)
    if not os.path.exists('model'):
        print('Creating new model')
        model = cnn.create_model()
        if not os.path.exists('dataset_labels.npy'):
            print('Generating dataset')
            label_dict = cnn.generate_dataset()
            with open('label_dict.pkl','wb') as f:
                pickle.dump(label_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('Training model')
        model = cnn.train_data(model, 'dataset.npy', 'dataset_labels.npy')
    else:
        model = models.load_model('model')
        print('Loaded model from data')
    print('Press ctrl+alt+p to predict drawing, or press ctrl+alt+q to quit')
    keyboard.wait('ctrl+alt+q')