import * as tf from '@tensorflow/tfjs';

// // Define a model for linear regression.
// const model = tf.sequential();
// model.add(tf.layers.dense({units: 1, inputShape: [1]}));

// model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// // Generate some synthetic data for training.
// const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
// const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

// // Train the model using the data.
// model.fit(xs, ys, {epochs: 100}).then(() => {
//   // Use the model to do inference on a data point the model hasn't seen before:
//   model.predict(tf.tensor2d([5], [1, 1]));
//   // Open the browser devtools to see the output
// });
// console.log(model.getWeights()[0])


const model = tf.sequential();
model.add(
  tf.layers.conv2d({
    inputShape: [28, 28, 1],
    kernelSize: 5,
    filters: 16,
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
    filters: 16,
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
model.add(
  tf.layers.dense({
    units: 10,
    activation: 'softmax',
    kernelInitializer: 'varianceScaling',
  }),
);
model.compile({
  optimizer: tf.train.sgd(0.01),
  loss: 'binaryCrossentropy',
  metrics: ['accuracy'],
});
// print the log
console.log(model.getWeights()[0].dataSync())
// console.log(model.getUserDefinedMetadata())