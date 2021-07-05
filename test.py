import flask
from flask import Flask,render_template
app=Flask(__name__)
@app.route('/')
def hello():
    return "hello world"
@ app.route('/index/<int:num>')
def form(num):
    return render_template('index.html',n=num)



if __name__=='__main__':
    app.run(debug=True)