var context = document.getElementById('sheet').getContext("2d");

function getColorAt(x, y){	
	var p = context.getImageData(x, y, 1, 1).data; 
	return p[3];
}

function getPixels(){
	var image = context.getImageData(0,0,context.canvas.width,context.canvas.height);
	var height = image.height;
	var width = image.width;
	var data = image.data;
	var pixels = []
	for(var i=0; i<data.length; i+=4){
		pixels.push(data[i+3]);
	}
	return {height:height, width:width, data:pixels};
}

function normalize(){
	var image = getPixels();
	scaledHt = Math.floor(image.height/10);
	scaledWd = Math.floor(image.width/10);
	var tmpscaled = [];
	
	var tempsum = 0;
	for(var i=0; i<image.data.length; i++){
		if(i%10 == 0){
			tmpscaled.push(tempsum);
			tempsum = 0;
		}
		tempsum += image.data[i];
	}
	var scaled = [];
	tempsum = Array.apply(null, Array(scaledHt)).map(function (x, i) { return 0; });
	var offset = 0;
	for(var i=0; i<tmpscaled.length; i++){
		tempsum[i%scaledHt] += tmpscaled[i];
		if(i != 0 && i%(10*scaledHt) == 0){
			scaled.push.apply(scaled, tempsum);
			tempsum = Array.apply(null, Array(scaledHt)).map(function (x, i) { return 0; });
		}
	}
	scaled.push.apply(scaled, tempsum);
	for(var i=0; i<scaled.length; i++){
		scaled[i] = scaled[i]/100;
	}
	return scaled;
}

function drawScaled(){
	pixels = normalize();
	console.log("np.argmax(predict([["+pixels.toString()+"]]))");
	var ctx = document.getElementById('scaled').getContext('2d');
	ctx.scale(10,10);
	var imgdata = ctx.getImageData(0,0, 28, 28);
	var imgdatalen = imgdata.data.length;
	for(var i=0;i<imgdatalen/4;i++){  //iterate over every pixel in the canvas
		imgdata.data[4*i] = 0;    // RED (0-255)
		imgdata.data[4*i+1] = 0;    // GREEN (0-255)
		imgdata.data[4*i+2] = 0;    // BLUE (0-255)
		imgdata.data[4*i+3] = pixels[i];  // APLHA (0-255)
	}
	ctx.putImageData(imgdata,0,0);
}