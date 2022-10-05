import * as tf from '@tensorflow/tfjs';
// import fs from 'fs';

// Creating a model
const model = tf.sequential();
model.add(
  tf.layers.dense({
    inputShape: 784,
    units: 1,
    activation: 'sigmoid',
    useBias: true
  }),
);
model.compile({
  optimizer: tf.train.sgd(0.01),
  loss: 'binaryCrossentropy',
  metrics: ['accuracy'],
});

// Receive json model from pytorch
import myJson from './lr_py.json' assert {type: 'json'};
let py_layers = Object.keys(myJson);
// let py_params = tf.tensor(Object.values(myJson))
// console.log(py_layers[0])
// console.log('Hi from python:\n',tf.tensor(myJson[py_layers[0]]))
// console.log(Object.keys(myJson))
// console.log('Hi from local tensorflow:\n',model.getWeights()[0])
let newWeights = [];
py_layers.forEach(item => {
  newWeights.push(tf.transpose(tf.tensor(myJson[item])))
})
console.log('New weights:\n',newWeights)
console.log('tfjs weights:\n', model.getWeights())
console.log('Before weights (example):\n', model.getWeights()[1].dataSync())

// set weights (Model Update)
if (newWeights.length == model.getWeights().length) {
  model.setWeights(newWeights)
}
console.log('Updated weights (example):\n', model.getWeights()[1].dataSync())


