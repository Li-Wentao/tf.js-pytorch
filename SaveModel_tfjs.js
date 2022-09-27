let out = []
let dict = new Object();
for (let i = 0; i < model.getWeights().length; i++) {
  dict = {'model': model.getWeights()[i],
          'params': model.getWeights()[i].dataSync()}
  out.push(dict);
}
let file = JSON.stringify(out);

// Save to local
fs.writeFile("./test.json", file, (err) => {
  if (err) {
  console.error(err);
  return;
    }
});
console.log("Data has been Written");