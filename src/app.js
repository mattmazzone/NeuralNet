import NeuralNetwork from './NeuralNetwork.js'

const network = new NeuralNetwork([2, 2, 1]);

const inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
];

const outputs = [
    [0],
    [1],
    [1],
    [1]
];

for (let i = 0; i < inputs.length; i++) {
    const out = network.forward(inputs[i])._data;
    const cost = network.cost(outputs[i])._data;
    console.log(`${inputs[i]} -> ${out} (${cost})`);
}

console.log();
for (let i = 0; i < 10000; i++) {
    network.backpropagate(inputs, outputs);
}

for (let i = 0; i < inputs.length; i++) {
    const out = network.forward(inputs[i])._data;
    const cost = network.cost(outputs[i])._data;
    console.log(`${inputs[i]} -> ${out} (${cost})`);
}
