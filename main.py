from flask import Flask, render_template, url_for, redirect, request
from model_processing import ModelProcessing
import os


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        f = request.files['img']
        img_dir = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(img_dir)
        out, size = process_input(img_dir)
        #print(img_dir)
        return render_template('display.html', img=img_dir, out=out)
    else:
        return render_template('capture.html')

def process_input(input_img):
    model = 'models/cnnv2.h5'
    mp = ModelProcessing(model, input_img)
    out = mp.breed_probabilities()
    size = mp.get_size()
    print(out)
    return out, size

if __name__ == '__main__':
    app.run(debug=True)
