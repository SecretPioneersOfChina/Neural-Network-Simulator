/**
 * nn.js — 轻量神经网络（纯 JS，无依赖）
 * 架构：784 → 128(ReLU) → 64(ReLU) → N(Softmax)
 */

class NeuralNetwork {
  constructor(inputSize, hiddenSizes, outputSize) {
    this.inputSize = inputSize;
    this.hiddenSizes = hiddenSizes;
    this.outputSize = outputSize;
    this.layers = [inputSize, ...hiddenSizes, outputSize];
    this.learningRate = 0.01;
    this._initWeights();
  }

  _initWeights() {
    this.weights = [];
    this.biases = [];
    for (let i = 0; i < this.layers.length - 1; i++) {
      const fan_in = this.layers[i];
      const fan_out = this.layers[i + 1];
      // He 初始化
      const scale = Math.sqrt(2.0 / fan_in);
      this.weights.push(
        Array.from({ length: fan_in }, () =>
          Array.from({ length: fan_out }, () => randn() * scale)
        )
      );
      this.biases.push(new Array(fan_out).fill(0));
    }
  }

  // 前向传播，返回每层激活值
  forward(input) {
    this._activations = [input];
    this._preActivations = [];
    let current = input;
    for (let l = 0; l < this.weights.length; l++) {
      const W = this.weights[l];
      const b = this.biases[l];
      const isLast = l === this.weights.length - 1;
      const z = matVec(W, current, b);
      this._preActivations.push(z);
      const a = isLast ? softmax(z) : relu(z);
      this._activations.push(a);
      current = a;
    }
    return current;
  }

  // 反向传播（交叉熵 + softmax 合并梯度）
  backward(target) {
    const L = this.weights.length;
    const dW = this.weights.map(w => w.map(row => row.map(() => 0)));
    const db = this.biases.map(b => b.map(() => 0));

    // 输出层梯度（交叉熵 + softmax）
    let delta = this._activations[L].map((a, i) => a - target[i]);

    for (let l = L - 1; l >= 0; l--) {
      const aIn = this._activations[l];
      // 累积梯度
      for (let i = 0; i < aIn.length; i++) {
        for (let j = 0; j < delta.length; j++) {
          dW[l][i][j] += aIn[i] * delta[j];
        }
      }
      for (let j = 0; j < delta.length; j++) {
        db[l][j] += delta[j];
      }
      if (l > 0) {
        // 传播到上一层（ReLU 梯度）
        const prevDelta = new Array(aIn.length).fill(0);
        const W = this.weights[l];
        for (let i = 0; i < aIn.length; i++) {
          for (let j = 0; j < delta.length; j++) {
            prevDelta[i] += W[i][j] * delta[j];
          }
          // ReLU 导数
          prevDelta[i] *= this._activations[l][i] > 0 ? 1 : 0;
        }
        delta = prevDelta;
      }
    }

    // 更新参数
    const lr = this.learningRate;
    for (let l = 0; l < L; l++) {
      for (let i = 0; i < this.weights[l].length; i++) {
        for (let j = 0; j < this.weights[l][i].length; j++) {
          this.weights[l][i][j] -= lr * dW[l][i][j];
        }
      }
      for (let j = 0; j < this.biases[l].length; j++) {
        this.biases[l][j] -= lr * db[l][j];
      }
    }
  }

  // 单样本训练，返回损失
  train(input, labelIndex) {
    const target = new Array(this.outputSize).fill(0);
    target[labelIndex] = 1;
    const out = this.forward(input);
    const loss = -Math.log(Math.max(out[labelIndex], 1e-9));
    this.backward(target);
    return loss;
  }

  predict(input) {
    const out = this.forward(input);
    let maxIdx = 0;
    for (let i = 1; i < out.length; i++) {
      if (out[i] > out[maxIdx]) maxIdx = i;
    }
    return { probabilities: out, predicted: maxIdx };
  }

  // 调整输出层大小（新增类别时）
  resizeOutput(newOutputSize) {
    if (newOutputSize === this.outputSize) return;
    const lastL = this.weights.length - 1;
    const prevSize = this.layers[this.layers.length - 2];
    const oldOut = this.outputSize;
    // 扩展权重
    const scale = Math.sqrt(2.0 / prevSize);
    for (let i = 0; i < prevSize; i++) {
      while (this.weights[lastL][i].length < newOutputSize) {
        this.weights[lastL][i].push(randn() * scale);
      }
    }
    while (this.biases[lastL].length < newOutputSize) {
      this.biases[lastL].push(0);
    }
    this.outputSize = newOutputSize;
    this.layers[this.layers.length - 1] = newOutputSize;
  }

  // 序列化 / 反序列化
  serialize() {
    return JSON.stringify({
      layers: this.layers,
      weights: this.weights,
      biases: this.biases,
      outputSize: this.outputSize
    });
  }

  static deserialize(json) {
    const d = JSON.parse(json);
    const nn = new NeuralNetwork(d.layers[0],
      d.layers.slice(1, -1), d.layers[d.layers.length - 1]);
    nn.weights = d.weights;
    nn.biases = d.biases;
    nn.outputSize = d.outputSize;
    return nn;
  }
}

// ——— 工具函数 ———
function randn() {
  // Box-Muller
  const u = 1 - Math.random();
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function relu(arr) {
  return arr.map(x => Math.max(0, x));
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

function matVec(W, x, b) {
  // W: [in × out], x: [in] → [out]
  return b.map((bj, j) => {
    let s = bj;
    for (let i = 0; i < x.length; i++) s += W[i][j] * x[i];
    return s;
  });
}
