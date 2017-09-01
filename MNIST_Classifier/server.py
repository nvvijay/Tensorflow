from flask import Flask, request, render_template
from mnistclient import predict, softmax

app = Flask(__name__)

@app.route('/')
def showDefault():
	return render_template("index.html")

@app.route('/<fname>')
def show(fname):
	return render_template(fname)

@app.route('/predict/<img>')
def calc(img):
	img = [[float(x) for x in img.split(",")]]
	if(len(img[0]) < 28*28):
		return "Wrong input dimentions. Expected ?, 728"
	return str(softmax(predict(img)).tolist())

@app.route('/prediction/', methods=['POST', 'GET'])
def run():
	if request.method == 'GET':
		img = request.args.get("image")
		img = [[float(x) for x in img.split(",")]]
		print(img)
		return str(predict(img))
	if request.method == 'POST':
		print("in run")
		img = request.form.get("image")
		if(img==None):
			return "No image given"
		img = [[float(x) for x in img.split(",")]]
		if(len(img[0]) < 28*28):
			return "Wrong input dimentions. Expected ?, 728"
		return str(predict(img))

if __name__ == '__main__':
	app.run(host='0.0.0.0')

