from flask import Flask

from flask import make_response, request, current_app
from flask_cors import CORS, cross_origin

import os
from fileio import *
app = Flask(__name__)
from trainer import  *
#@app.route("/")
#def hello():
#    return "hello"
#@app.route('/user/<username>')
#def show_user_profile(username):
    # show the user profile for that user
#    return 'User %s' % username

@app.route('/upload', methods=['POST', 'GET'])
@cross_origin()
def upload():

    error = None
    print("Upload called")
    classtype = request.form['classtype']
    classvar = request.form['class']
    mode = request.form['mode']
    print("classtype = {0}, class={1}, mode = {2}".format(classtype,classvar,mode))

    numberofclasses=5

    fio=fileio(classtype)

    if 'file' not in request.files:
        print('no file')
    else :

        print(' file found')
        file = request.files['file']



        if( mode=='1'):
            print(' mode 1 ')
            filename=fio.WriteClassifyWav(file )
            trainer= mainTrainer(numberofclasses,classtype)
            rval = trainer.classify(filename)
            print ("Live training Result {0}".format(rval))
            return str(rval[0])
        else :
            fio.WriteNextWav(file, classtype, classvar)

        #file.save(os.getcwd()+"/abc.wav")
    return ""


if __name__ == "__main__":
    app.run()