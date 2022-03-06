import os
import json
import random

from PIL import Image
from wekzeug.utils import secure_filename
from flask import Flask, request, render_template

import Barbershop
import align_face

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './unprocessed'
app.comfig['RESULT_FOLDER'] = './output'
cur_path = os.path.abspath(os.getcwd())
base_dir = os.environ.get('BASE_DIR','')
print("base_dir :" + base_dir)
model = Barbershop()


@app.route(f"{base_dir}/v1/index", methods = ['GET', 'POST'])
def home():
    if not os.path.exists(app.config['UPLOAD_FLODER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    if request.method == 'GET':
        code_url = generate_code()
        return render_template('home.html', base_dir = base_dir, param_code = code_url)

    if request.method == 'POST':
        if 'image' not in request.files:
            json_obj = {'image_url' : '', 'blend' : '文件上传失败'}
            return json.dumps(json_obj)
        image_file = request.files['image']
        
        if image_file.filename == "":
            json_obj = {'image_url' : '', 'blend' : '文件上传为空'}
            return json.dumps(json_obj)
        
        if image_file and is_allowed_file(image_file.filename):
            try:
                filename = generate_filename(image_file.filename)
                filePath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image_file.save(filePath)
                return gen_img(filename)
            except Exception:
                json_obj = {'image_url' : '' : 'blend' : '后台异常'}
                return json.dumps(json_obj)

def is_allowed_file(filename):
    VALID_EXTENSIONS = ['png', 'jpg', 'jpeg']
    is_valid_ext = filename.rsplit('.', 1)[1].lower() in VALID_EXTENSIONS
    return '.' in filename and is_valid_ext

def generate_filename(filename):
    LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    ext = filename.split(".")[-1]
    random_indexes = [random.randint(0, len(LETTERS) - 1) for _ in range(10)]
    random_chars = "".join([LETTERS[index] for index in random_indexes])
    new_name = "{name}.{extension}".format(name=random_chars, extension=ext)
    return secure_filename(new_name)

def gen_img(filename):
    print('filename:'+filename)
    original_img_url = url_for('image', filename)
    original_img_path = os.path.join(
        app.config['UPLOAD_FOLDER'], filename)
    result = []

    try:
        out1, output_path = align_face(output_dir)



app.run('0.0.0.0')
