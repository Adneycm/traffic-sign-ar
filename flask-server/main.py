from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    return {"Home" : ["Welcome", "to", "the", "home", "page"]}

@app.route("/about")
def about():
    return {"About" : ["Welcome", "to", "the", "about", "page"]}


@app.route("/video-input", methods=["POST"])
def handle_video_upload():
  try:
    video_file = request.files["video"]  # Access the video file from the request
    # You can now save the video file to your server storage or perform other processing
    return {"message": "Video uploaded successfully!"}, 200
  except Exception as e:
    return {"error": str(e)}, 400



if __name__=='__main__':
    app.run(debug=True)