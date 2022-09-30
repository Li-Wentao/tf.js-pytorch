import * as tf from '@tensorflow/tfjs';
import fs from 'fs';

// Creating a model
const model = tf.sequential();
model.add(
  tf.layers.dense({
    inputShape: 28,
    units: 28,
    activation: 'sigmoid',
    // useBias: true
  }),
);
model.compile({
  optimizer: tf.train.sgd(0.01),
  loss: 'binaryCrossentropy',
  metrics: ['accuracy'],
});

// console.log(model.getWeights())
let out = []
let dict = new Object();
for (let i = 0; i < model.getWeights().length; i++) {
  dict = {'model': model.getWeights()[i],
          'params': model.getWeights()[i].dataSync()}
  out.push(dict);
}
let tf_model = JSON.stringify(out);

// Save to local
fs.writeFile("./lr_tf.json", tf_model, (err) => {
  if (err) {
  console.error(err);
  return;
    }
});
console.log("Model has been transmitted to Python");
