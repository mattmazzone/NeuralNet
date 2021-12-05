
const width = 1000, height = 1000;

const neuronRadius = 25;
const neuronDiameter = neuronRadius * 2;

const network = new NeuralNetwork([2, 2, 1]);

const deltaX = (width - (network.layers.length * neuronDiameter)) / (network.layers.length + 1);

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

function setup() {
    createCanvas(width, height);
}

function draw() {
    background(33, 37, 47);
    fill(255);

    network.backpropagate(inputs, outputs);
    const out = network.forward(inputs[1]);
    const cost = network.cost(outputs[1]);

    fill(0, 102, 153);
    textSize(32);
    text(cost._data[0], 200, 200);

    const positions = [];
    for (let i = 0; i < network.layers.length; i ++) {
        let deltaY = (height - (network.layers[i] * neuronDiameter)) / (network.layers[i] + 1);
        let layer = [];
        for (let j = 0; j < network.layers[i]; j ++){
            const x = (i * neuronDiameter) + neuronRadius + (deltaX * (i + 1));
            const y = (j * neuronDiameter) + neuronRadius + (deltaY * (j + 1));
            layer.push([x, y]);
        }
        positions.push(layer);
    }

    fill(255);
    stroke(0);
    for (let l = 0; l < positions.length; l++) {
        for (let n = 0; n < positions[l].length; n++) {
            const current = positions[l][n];
            strokeWeight(4);
            if (l < positions.length - 1) {
                for (let nl = 0; nl < positions[l + 1].length; nl++) {
                    stroke(neuronColor(l, nl, n));
                    const next = positions[l + 1][nl];
                    line(current[0], current[1], next[0], next[1]);
                }
            }
            stroke(0);
            strokeWeight(2);
            fill(network.activations[l]._data[n] * 255);
            circle(current[0], current[1], neuronDiameter);
        }
    }
}

function neuronColor(l, nl, n) {
    const currentWeight = network.weights[l]._data[nl][n];
    if (currentWeight > 0) {
        const normalizedWeight = currentWeight / network.maxWeight();
        return lerpColor(color(80), color(137, 202, 120), normalizedWeight);
    } else {
        const normalizedWeight = 1 - currentWeight / network.minWeight();
        return lerpColor(color(239, 62, 93), color(80), normalizedWeight);
    }
}
