import math from './math.js';

const sigmoid = x => 1 / (1 + math.exp(-x));
const sigmoidInverse = x => math.log(x / (1 - x));
const sigmoidPrime = x => {
    let sig = sigmoid(x);
    return sig * (1 - sig);
}

const diffSquares = (output, expected) => math.map(math.subtract(expected, output), x => math.pow(x, 2));
const diffSquaresPrime = (output, expected) => math.subtract(expected, output);

function randomEvenFn(range) {
    return _ => Math.random() * 2 * range - range;
}

export default class NeuralNetwork {

    constructor(layers, activationFn = sigmoid, activationInverseFn = sigmoidInverse, activationPrimeFn = sigmoidPrime, costFn = diffSquares, costPrimeFn = diffSquaresPrime) {
        this.layers = layers;

        this.weights = [...Array(layers.length - 1)]
            .map((_, i) => math.map(math.zeros(layers[i + 1], layers[i]), randomEvenFn(5)));
        this.biases = [...Array(layers.length - 1)]
            .map((_, i) => math.map(math.zeros(layers[i + 1]), randomEvenFn(5)));

        this.activationFn = activationFn;
        this.activationInverseFn = activationInverseFn;
        this.activationPrimeFn = activationPrimeFn;
        this.costFn = costFn;
        this.costPrimeFn = costPrimeFn;
    }

    forward(input) {
        // Turn input into matrix
        let next = math.matrix(input);

        // Compute first layer weighted input and activation
        this.weightedInputs = [math.map(next, this.activationInverseFn)];
        this.activations = [next];

        // Propagate forward through network
        for (let i = 0; i < this.layers.length - 1; i++) {

            // Weighted inputs of layer
            // z[l] = w[l] * a[l-1] + b[l]
            // Store this value because it is needed to compute the gradients
            let weightedInput = math.add(math.multiply(this.weights[i], next), this.biases[i]);
            this.weightedInputs.push(weightedInput);

            // Activations of layer
            // a[l] = activation(z[l])
            next = math.map(weightedInput, this.activationFn);
            this.activations.push(next);

        }

        // Return output
        return next;
    }

    gradient(input, expected) {
        // Compute output
        const output = this.forward(input);

        // ** is used for Hadamard product (element-wise multiplication as opposed to matrix multiplication)
        // Compute error terms of last layer
        // d[l] = gradient(C, a) ** activation'(z[l])
        let costGradient = this.costPrimeFn(output, expected);
        let activationDerivative = math.map(this.weightedInputs[this.weightedInputs.length - 1], this.activationPrimeFn);
        let last = math.dotMultiply(costGradient, activationDerivative);
        this.errorTerms = [last];

        // Compute error terms for each layer before the last
        for (let l = this.weights.length - 1; l >= 0; l--) {
            // d[l] = (transpose(w[l+1]) * d[l+1]) ** activation'(z[l])
            last = math.dotMultiply(math.multiply(math.transpose(this.weights[l]), last), math.map(this.weightedInputs[l], this.activationPrimeFn));
            this.errorTerms.unshift(last);
        }

        // Compute gradient of weights and biases
        this.weightsGradient = [];
        this.biasesGradient = [];
        for (let l = 0; l < this.weights.length; l++) {
            // (dC/db)[l] = d[l]
            // NOTE: The l index on the error term is l+1 in our case since the
            // error term array will always have one more element than the biases array
            this.biasesGradient[l] = this.errorTerms[l + 1];

            // dC/dw = a_in * d_out
            // (dC/dw[j,k])[l] = (a[k])[l-1] * (d[j])[l]
            // NOTE: The l-1 index on the activation is simply l in our case since the
            // activation array will always have one more element than the weight array
            let gradientMatrix = [];
            for (let j = 0; j < this.weights[l]._size[0]; j++) {
                let column = [];
                for (let k = 0; k < this.weights[l]._size[1]; k++) {
                    // dC/dw[j,k] = a[k] * d[j]
                    let gradient = this.activations[l]._data[k] * this.errorTerms[l + 1]._data[j];
                    column.push(gradient);
                }
                gradientMatrix.push(column);
            }
            this.weightsGradient.push(math.matrix(gradientMatrix));
        }
    }

    cost(expected) {
        if (!this.activations)
            throw new Error('Must provide an input prior to calculating cost');

        return this.costFn(this.activations[this.activations.length - 1], expected);
    }

    backpropagate(inputs, expected) {
        // Compute total gradient sums
        let totalWeightGradient = [];
        let totalBiasGradient = [];
        for (let i = 0; i < inputs.length; i++) {
            this.gradient(inputs[i], expected[i]);
            if (i === 0) {
                totalWeightGradient = [...this.weightsGradient];
                totalBiasGradient = [...this.biasesGradient];
            } else {
                totalWeightGradient = totalWeightGradient.map((tw, i) => math.add(tw, this.weightsGradient[i]));
                totalBiasGradient = totalBiasGradient.map((tb, i) => math.add(tb, this.biasesGradient[i]));
            }
        }

        // Divide by number of entries in batch
        totalWeightGradient = totalWeightGradient.map(tw => math.divide(tw, inputs.length));
        totalBiasGradient = totalBiasGradient.map(tb => math.divide(tb, inputs.length));

        // Apply total gradient to weights and biases
        this.weights = this.weights.map((w, i) => math.add(w, totalWeightGradient[i]));
        this.biases = this.biases.map((b, i) => math.add(b, totalBiasGradient[i]));
    }

}
