import * as tf from '@tensorflow/tfjs';
import fs from 'fs';
// import * as tf from 'tfjs-onnx';
// import fetch from "node-fetch";

// Creating a model
const model = tf.sequential();
model.add(
  tf.layers.conv2d({
    inputShape: [30, 30, 1], // picture size
    kernelSize: 3,          
    filters: 5,              // out_channels in pytorch
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling',
  }),
);
model.add(
  tf.layers.maxPool2d({
    poolSize: [2, 2],
    strides: [2, 2],
  }),
);
model.add(
  tf.layers.conv2d({
    kernelSize: 5,
    filters: 5,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling',
  }),
);
model.add(
  tf.layers.maxPool2d({
    poolSize: [2, 2],
    strides: [2, 2],
  }),
);
model.add(tf.layers.flatten());
// model.add(
//   tf.layers.dense({
//     units: 10,
//     activation: 'softmax',
//     kernelInitializer: 'varianceScaling',
//   }),
// );
model.compile({
  optimizer: tf.train.sgd(0.01),
  loss: 'binaryCrossentropy',
  metrics: ['accuracy'],
});

  
model.compile({loss: 'categoricalCrossentropy', optimizer: 'sgd'});

// print the log
// console.log(JSON.stringify(model.getWeights()[0].dataSync()));
// console.log(model.getWeights()[0].dataSync())
// console.log(model.getWeights()[0].print())
let params = []
for (let i = 0; i < model.getWeights().length; i++) {
  params.push(model.getWeights()[i].dataSync());
}
let test_var = JSON.stringify(params);
console.log(params[2])
// let test_var = JSON.stringify(model.getWeights()[0]);
// console.log(test_var)


// Save to local
fs.writeFile("./test.json", test_var, (err) => {
  if (err) {
  console.error(err);
  return;
    }
});
console.log("Data has been Written");

// // var modelUrl = 'models/bvlc_googlenet/model.onnx';
// var modelUrl = 'models/squeezenet/model.onnx';
// // Initialize the tf.model
// var test_model = new tf.onnx.loadModel(modelUrl);
// model.save('test_model');
// console.log(model.toJSON())