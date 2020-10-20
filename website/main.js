const letters = 'RraGg3Wfy';
const width = 6;
const height = 13;
const pixelSize = 17;

var myOnnxSession;
var ctxOutput;
var ctxs = {};

let glyphX = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[1,0,0,0,0,0],[1,0,0,0,0,0],[1,1,1,0,0,0],[1,0,0,1,0,0],[1,0,0,1,0,0],[1,1,1,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]];

//var glyphs = {"3":glyph3,"R":glyphR};

//letters.split('').forEach( (letter) =>
//  glyphs[letter] = JSON.parse(JSON.stringify(glyphX))
//);

glyphs = {"3":[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,1,1,1,0,0],[1,0,0,0,1,0],[0,0,0,0,1,0],[0,1,1,1,0,0],[0,0,0,0,1,0],[1,0,0,0,1,0],[0,1,1,1,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]],"R":[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[1,1,1,1,0,0],[1,0,0,0,1,0],[1,0,0,0,1,0],[1,1,1,1,0,0],[1,0,0,0,1,0],[1,0,0,0,1,0],[1,0,0,0,1,0],[0,0,0,0,0,0],[0,0,0,0,0,0]],"r":[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[1,0,1,1,0,0],[1,1,0,0,1,0],[1,0,0,0,0,0],[1,0,0,0,0,0],[1,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]],"a":[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,1,1,1,0,0],[0,0,0,0,1,0],[0,1,1,1,1,0],[1,0,0,0,1,0],[1,0,0,0,1,0],[0,1,1,1,1,0],[0,0,0,0,0,0],[0,0,0,0,0,0]],"G":[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,1,1,1,0,0],[1,0,0,0,1,0],[1,0,0,0,0,0],[1,0,1,1,0,0],[1,0,0,0,1,0],[1,0,0,0,1,0],[0,1,1,1,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]],"g":[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,1,1,1,0,0],[1,0,0,0,1,0],[1,0,0,0,1,0],[0,1,1,1,1,0],[0,0,0,0,1,0],[0,1,1,1,1,0]],"W":[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[1,0,0,0,1,0],[1,0,0,0,1,0],[1,0,0,0,1,0],[1,0,0,0,1,0],[1,0,0,0,1,0],[1,0,1,0,1,0],[0,1,0,1,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]],"f":[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,1,1,0,0],[0,1,0,0,1,0],[0,1,0,0,0,0],[1,1,1,0,0,0],[0,1,0,0,0,0],[0,1,0,0,0,0],[0,1,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]],"y":[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[1,0,0,0,1,0],[1,0,0,0,1,0],[1,0,0,0,1,0],[0,1,1,1,1,0],[0,0,0,0,1,0],[0,0,1,1,1,0]]};

function randomGlyph() {
  let result = [];
  for ( var y = 0 ; y < height; y++ ) {
    result[y] = [];
    for ( var x = 0 ; x < width; x++ ) {
      result[y][x] = 0;
      if (Math.random() > 0.5)
	result[y][x] = 0;
    }
  }
  return result;
}

function drawEditor( ctx, glyph ) {
  ctx.fillStyle = "#FEFEFE";  
  ctx.clearRect(0, 0, width * pixelSize + 1, height * pixelSize + 1);

  const opacity = 0.1;
  ctx.strokeStyle = "rgba(0, 0, 0, " + opacity + ")";
  ctx.lineWidth = '1px';
  
  for( var x=0; x<=width; x++ ) {
    ctx.beginPath();
    ctx.moveTo(x * pixelSize, 0);
    ctx.lineTo(x * pixelSize, height * pixelSize);
    ctx.stroke();
  }
  for( var y=0; y<=height; y++ ) {
    ctx.beginPath();
    ctx.moveTo(0, y * pixelSize);
    ctx.lineTo(width * pixelSize, y * pixelSize);
    ctx.stroke();
  }

  ctx.fillStyle = "#000";
  for( var y=0; y<height; y++ ) {
    for( var x=0; x<width; x++ ) {
      if (glyph[y][x] == 1) {
	ctx.fillRect(x*pixelSize, y*pixelSize,
		     pixelSize, pixelSize);
      }
    }
  }
}

function relMouseCoords(event){
  var totalOffsetX = 0;
  var totalOffsetY = 0;
  var canvasX = 0;
  var canvasY = 0;
  var currentElement = event.target;
  
  do{
    totalOffsetX += currentElement.offsetLeft - currentElement.scrollLeft;
    totalOffsetY += currentElement.offsetTop - currentElement.scrollTop;
  }
  while(currentElement = currentElement.offsetParent)
  
  canvasX = event.pageX - totalOffsetX;
  canvasY = event.pageY - totalOffsetY;
  
  return {x:canvasX, y:canvasY};
}

var drawingMode = 0;

function mousedown(e) {
  let canvas = this;
  let p = relMouseCoords(e);
  p.x /= pixelSize;
  p.y /= pixelSize;
  p.x = Math.floor(p.x);
  p.y = Math.floor(p.y);

  let letter = canvas.getAttribute('letter');
  
  drawingMode = 1 - glyphs[letter][p.y][p.x];
  glyphs[letter][p.y][p.x] = drawingMode;

  drawEditor( ctxs[letter], glyphs[letter] );

  runModel();

  return false;
}

function mousemove(e) {
  if (e.buttons) {
    let canvas = this;
    let p = relMouseCoords(e);
    p.x /= pixelSize;
    p.y /= pixelSize;
    p.x = Math.floor(p.x);
    p.y = Math.floor(p.y);

    let letter = canvas.getAttribute('letter');
    glyphs[letter][p.y][p.x] = drawingMode;
    drawEditor( ctxs[letter], glyphs[letter] );
    runModel();

    return false;
  }
  
}

function makeEditor(letter) {
  var editor = document.createElement("div");
  editor.classList.add("editor");
  
  var canvas = document.createElement("canvas");

  var label = document.createElement("span");
  label.appendChild( document.createTextNode(letter) );
  label.classList.add("label");
  
  ctxs[letter] = canvas.getContext("2d");
  canvas.addEventListener("mousemove",mousemove,false);
  canvas.addEventListener("mousedown",mousedown,false);
  
  canvas.width = width * pixelSize + 1;
  canvas.height = height * pixelSize + 1;
  canvas.setAttribute('letter', letter);
  editor.appendChild(label);
  editor.appendChild(canvas);
  document.getElementById('main').prepend(editor);

  drawEditor( ctxs[letter], glyphs[letter] );
}

const spacing = 5;
const scale = 3;

function drawOutput(tensor) {
  const glyphCount = tensor.dims[0];

  const rounded = new Uint32Array(tensor.data.map( (x) => {
    x = Math.pow(x,3);
    x = Math.floor(x * 255);
    return x;
  }));
  
  const data = new Uint32Array(rounded.map( (x) => {
    if (x < 0) x = 0;
    if (x > 255) x = 255;
    x = 255 - x;
    return x | (x << 8) | (x << 16) | (255 << 24);
  }));

  ctxOutput.imageSmoothingEnabled = false;
    
  for( var i = 0; i < glyphCount; i ++ ) {
    let b = data.slice(i*width*height, (i+1)*width*height).buffer;
    //console.log(b);
    var glyph = new ImageData(new Uint8ClampedArray(b), width, height);
    (function(index) {
      let x = index % 16;
      let y = Math.floor(index / 16);
      createImageBitmap(glyph).then(renderer =>
	ctxOutput.drawImage(renderer,
			    x*(scale*width+spacing),
			    y*(scale*height+spacing),
			    scale*width, scale*height)
      );
    })(i);
  }
}

function makeOutput() {
  var loading = document.getElementById("loader");
  if (loading)
    loading.remove();
  
  var output = document.createElement("div");
  output.classList.add("output");
  var canvas = document.createElement("canvas");

  canvas.width = (scale * width + spacing) * 16;
  canvas.height = (scale * height + spacing) * 6;
  
  output.appendChild(canvas); 
  document.getElementById('main').prepend(output);

  ctxOutput = canvas.getContext("2d");
}

let timer = undefined;
let throttle = 50;
function runModel() {
  if (timer === undefined) {
    timer = setTimeout(runModelImmediately, throttle);
  } else {
    clearTimeout(timer);
    timer = setTimeout(runModelImmediately, throttle);
  }
}

function runModelImmediately() {
  timer = undefined;
  
  const glyphCount = 128 - 32;
  let xs = letters.split('').map( (letter) => glyphs[letter] ).flat().flat();
  const forms = new Float32Array(xs);

  let allForms = new Float32Array(xs.length * (glyphCount));
  allForms.set(forms);
  for( var i = 1; i < glyphCount; i++ ) {
    allForms.copyWithin( i * xs.length, 0, xs.length );
  }

  const codepoints = new Int32Array(glyphCount);
  for( var i = 0; i < glyphCount; i++ ) {
    codepoints[i] = i;
  }
  
  const inputs = [
    new Tensor(codepoints, "int32", [glyphCount]),
    new Tensor(allForms, "float32",
	       [glyphCount, letters.length, height, width]),
  ];
  
  myOnnxSession.run(inputs).then((output) => {
    // consume the output
    const outputTensor = output.values().next().value;
    window.tensor = outputTensor;
    console.log("Model ran!");
    drawOutput(outputTensor);
  });
}

window.addEventListener('load', function () {
  letters.split('').forEach( (letter) => {
    makeEditor(letter);
  });

  // create a session
  myOnnxSession = new onnx.InferenceSession();
  
  // load the ONNX model file
  myOnnxSession.loadModel("./model.onnx").then(() => {
    makeOutput();
    runModel();
  });
   
});
