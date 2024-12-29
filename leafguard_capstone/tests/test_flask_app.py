from flask import Flask

server = Flask(__name__)

@server.route("/")
def hello():
    return "Flask is working!"

if __name__ == "__main__":
    server.run(host="127.0.0.1", port=8050, debug=True)
