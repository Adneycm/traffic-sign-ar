from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    return {"Home" : ["Welcome", "to", "the", "home", "page"]}

@app.route("/about")
def about():
    return {"About" : ["Welcome", "to", "the", "about", "page"]}

if __name__=='__main__':
    app.run(debug=True)