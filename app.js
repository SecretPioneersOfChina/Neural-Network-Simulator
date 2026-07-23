/**
 * app.js — 主应用逻辑
 */

// ===== 全局状态 =====
const CANVAS_SIZE = 280;   // 显示大小
const PIXEL_SIZE  = 28;    // 神经网络输入大小

// 默认 10 类：数字 0-9
let labels = ['0','1','2','3','4','5','6','7','8','9'];
let nn = new NeuralNetwork(784, [128, 64], labels.length);

let customSamples = [];   // { pixels: Float32Array(784), label: number }
let trainingHistory = []; // 损失记录
let isTraining = false;
let trainInterval = null;
let totalEpoch = 0;

// ===== 画板 =====
const canvas    = document.getElementById('drawCanvas');
const ctx       = canvas.getContext('2d');
let drawing     = false;
let lastX = 0, lastY = 0;

function initCanvas() {
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 18;
  ctx.lineCap   = 'round';
  ctx.lineJoin  = 'round';
}

canvas.addEventListener('mousedown', e => {
  drawing = true;
  [lastX, lastY] = getPos(e);
});
canvas.addEventListener('mousemove', e => {
  if (!drawing) return;
  const [x, y] = getPos(e);
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(x, y);
  ctx.stroke();
  [lastX, lastY] = [x, y];
});
canvas.addEventListener('mouseup',   () => drawing = false);
canvas.addEventListener('mouseleave',() => drawing = false);

// 触摸支持
canvas.addEventListener('touchstart', e => {
  e.preventDefault();
  drawing = true;
  [lastX, lastY] = getTouchPos(e);
});
canvas.addEventListener('touchmove', e => {
  e.preventDefault();
  if (!drawing) return;
  const [x, y] = getTouchPos(e);
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(x, y);
  ctx.stroke();
  [lastX, lastY] = [x, y];
});
canvas.addEventListener('touchend', () => drawing = false);

function getPos(e) {
  const r = canvas.getBoundingClientRect();
  return [e.clientX - r.left, e.clientY - r.top];
}
function getTouchPos(e) {
  const r = canvas.getBoundingClientRect();
  const t = e.touches[0];
  return [t.clientX - r.left, t.clientY - r.top];
}

// ===== 像素提取 =====
function getPixels() {
  // 把 280×280 的画布缩放到 28×28
  const offCanvas = document.createElement('canvas');
  offCanvas.width  = PIXEL_SIZE;
  offCanvas.height = PIXEL_SIZE;
  const offCtx = offCanvas.getContext('2d');
  offCtx.drawImage(canvas, 0, 0, PIXEL_SIZE, PIXEL_SIZE);
  const imgData = offCtx.getImageData(0, 0, PIXEL_SIZE, PIXEL_SIZE);
  const pixels = new Float32Array(PIXEL_SIZE * PIXEL_SIZE);
  for (let i = 0; i < pixels.length; i++) {
    // 取亮度（灰度均值 / 255）
    const base = i * 4;
    pixels[i] = (imgData.data[base] + imgData.data[base+1] + imgData.data[base+2]) / (3 * 255);
  }
  return pixels;
}

// ===== 清空画板 =====
document.getElementById('btnClear').addEventListener('click', () => {
  initCanvas();
  clearResult();
});

// ===== 识别 =====
document.getElementById('btnPredict').addEventListener('click', predict);

function predict() {
  const pixels = getPixels();
  const maxVal = Math.max(...pixels);
  if (maxVal < 0.05) {
    showToast('画板为空，请先手写内容再识别');
    return;
  }
  if (customSamples.length === 0 && totalEpoch === 0) {
    showToast('⚠️ 网络尚未训练，识别结果为随机猜测');
  }
  const { probabilities, predicted } = nn.predict(Array.from(pixels));
  showResult(predicted, probabilities);
  updateNetworkViz(Array.from(pixels), probabilities);
}

function showResult(predicted, probs) {
  const predictedLabel = labels[predicted] !== undefined ? labels[predicted] : '?';
  document.getElementById('resultLabel').textContent = predictedLabel;

  const bars = document.getElementById('probBars');
  bars.innerHTML = '';

  // 按概率降序排列，取前 min(labels.length, 10) 项
  const sorted = probs
    .map((p, i) => ({ p, i }))
    .sort((a, b) => b.p - a.p)
    .slice(0, Math.min(labels.length, 10));

  const bestIdx = sorted[0].i; // 概率最高的原始类别下标

  sorted.forEach(({ p, i }) => {
    const pct = (p * 100).toFixed(1);
    const labelText = labels[i] !== undefined ? labels[i] : String(i);
    const isBest = (i === bestIdx);
    bars.innerHTML += `
      <div class="prob-row${isBest ? ' best' : ''}">
        <span class="prob-label">${labelText}</span>
        <div class="prob-bar-bg">
          <div class="prob-bar" style="width:${pct}%"></div>
        </div>
        <span class="prob-pct">${pct}%</span>
      </div>`;
  });
}

function clearResult() {
  document.getElementById('resultLabel').textContent = '?';
  document.getElementById('probBars').innerHTML = '';
}

// ===== 添加自定义训练样本 =====
document.getElementById('btnAddSample').addEventListener('click', () => {
  const labelInput = document.getElementById('customLabel').value.trim();
  if (!labelInput) {
    alert('请先输入符号名称');
    return;
  }
  // 检查画板是否全黑（未画任何内容）
  const checkPixels = getPixels();
  const maxVal = Math.max(...checkPixels);
  if (maxVal < 0.05) {
    alert('画板为空！请先在画板上画出符号再添加样本。');
    return;
  }
  // 判断是否新类别
  let labelIdx = labels.indexOf(labelInput);
  if (labelIdx === -1) {
    labels.push(labelInput);
    labelIdx = labels.length - 1;
    nn.resizeOutput(labels.length);
    updateLabelList();
    renderLayerEditor();
    updateHeaderDesc();
  }
  const pixels = getPixels();
  customSamples.push({ pixels: Array.from(pixels), label: labelIdx });
  updateSampleCount();
  showToast(`已添加 "${labelInput}" 样本（共 ${customSamples.filter(s=>s.label===labelIdx).length} 个）`);
  initCanvas();
});

function updateLabelList() {
  const el = document.getElementById('labelList');
  el.innerHTML = labels.map((l, i) => {
    const cnt = customSamples.filter(s => s.label === i).length;
    return `<span class="label-badge">${l}${cnt > 0 ? ` (${cnt})` : ''}</span>`;
  }).join('');
}

function updateSampleCount() {
  document.getElementById('sampleCount').textContent =
    `自定义样本：${customSamples.length} 个`;
  updateLabelList();
}

// ===== 删除类别 =====
document.getElementById('btnDeleteLabel').addEventListener('click', () => {
  const labelInput = document.getElementById('customLabel').value.trim();
  const idx = labels.indexOf(labelInput);
  if (idx === -1) { alert('类别不存在'); return; }
  if (idx < 10 && labels.length > 10) {
    alert('默认数字类别 0-9 不可删除');
    return;
  }
  labels.splice(idx, 1);
  customSamples = customSamples
    .filter(s => s.label !== idx)
    .map(s => ({ ...s, label: s.label > idx ? s.label - 1 : s.label }));
  nn = new NeuralNetwork(784, [...editingHiddenSizes], labels.length);
  updateLabelList();
  updateSampleCount();
  renderLayerEditor();
  updateHeaderDesc();
  showToast(`类别 "${labelInput}" 已删除，网络已重置`);
});

// ===== 训练控制 =====
document.getElementById('btnTrain').addEventListener('click', toggleTraining);

function toggleTraining() {
  if (isTraining) {
    stopTraining();
  } else {
    startTraining();
  }
}

function startTraining() {
  if (customSamples.length === 0) {
    showToast('请先添加自定义训练样本，或使用内置 MNIST 权重');
    return;
  }
  isTraining = true;
  document.getElementById('btnTrain').textContent = '⏹ 停止训练';
  document.getElementById('btnTrain').classList.add('active');
  const batchSize = 8;
  const lr = parseFloat(document.getElementById('lrInput').value) || 0.01;
  nn.learningRate = lr;

  trainInterval = setInterval(() => {
    let epochLoss = 0;
    // 随机小批量
    for (let b = 0; b < batchSize; b++) {
      const s = customSamples[Math.floor(Math.random() * customSamples.length)];
      epochLoss += nn.train(s.pixels, s.label);
    }
    const avgLoss = epochLoss / batchSize;
    totalEpoch++;
    trainingHistory.push(avgLoss);
    if (trainingHistory.length > 200) trainingHistory.shift();
    updateLossChart();
    document.getElementById('epochCount').textContent = `迭代：${totalEpoch}`;
    document.getElementById('lossDisplay').textContent = `损失：${avgLoss.toFixed(4)}`;
  }, 50);
}

function stopTraining() {
  isTraining = false;
  clearInterval(trainInterval);
  document.getElementById('btnTrain').textContent = '▶ 开始训练';
  document.getElementById('btnTrain').classList.remove('active');
}

// ===== 学习率调整 =====
document.getElementById('lrInput').addEventListener('input', e => {
  nn.learningRate = parseFloat(e.target.value) || 0.01;
  document.getElementById('lrDisplay').textContent = e.target.value;
});

// ===== 重置网络 =====
document.getElementById('btnReset').addEventListener('click', () => {
  if (!confirm('重置会清除训练权重，确定？')) return;
  stopTraining();
  nn = new NeuralNetwork(784, [...editingHiddenSizes], labels.length);
  trainingHistory = [];
  totalEpoch = 0;
  updateLossChart();
  document.getElementById('epochCount').textContent = '迭代：0';
  document.getElementById('lossDisplay').textContent = '损失：-';
  renderLayerEditor();
  updateHeaderDesc();
  showToast('网络已重置');
});

// ===== 保存 / 加载权重 =====
document.getElementById('btnSave').addEventListener('click', () => {
  const data = {
    weights: nn.serialize(),
    labels: labels,
    customSamples: customSamples
  };
  const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'nn_weights.json';
  a.click();
});

document.getElementById('btnLoad').addEventListener('click', () => {
  document.getElementById('fileInput').click();
});

document.getElementById('fileInput').addEventListener('change', e => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = ev => {
    try {
      const data = JSON.parse(ev.target.result);
      nn = NeuralNetwork.deserialize(data.weights);
      labels = data.labels || labels;
      customSamples = data.customSamples || [];
      // 恢复编辑器状态
      editingHiddenSizes = [...nn.hiddenSizes];
      renderLayerEditor();
      updateHeaderDesc();
      updateLabelList();
      updateSampleCount();
      showToast('权重加载成功！');
    } catch {
      alert('文件格式错误');
    }
  };
  reader.readAsText(file);
  e.target.value = '';
});

// ===== 损失曲线 =====
const lossCanvas = document.getElementById('lossCanvas');
const lossCtx    = lossCanvas.getContext('2d');

function updateLossChart() {
  const w = lossCanvas.width, h = lossCanvas.height;
  lossCtx.clearRect(0, 0, w, h);
  if (trainingHistory.length < 2) return;

  const maxLoss = Math.max(...trainingHistory, 0.1);
  const minLoss = Math.min(...trainingHistory);
  const range = maxLoss - minLoss || 1;

  // 背景格线
  lossCtx.strokeStyle = 'rgba(255,255,255,0.08)';
  lossCtx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = (h - 20) * (i / 4) + 10;
    lossCtx.beginPath();
    lossCtx.moveTo(0, y);
    lossCtx.lineTo(w, y);
    lossCtx.stroke();
  }

  // 曲线
  lossCtx.strokeStyle = '#4fc3f7';
  lossCtx.lineWidth = 2;
  lossCtx.beginPath();
  trainingHistory.forEach((v, i) => {
    const x = (i / (trainingHistory.length - 1)) * w;
    const y = h - 10 - ((v - minLoss) / range) * (h - 20);
    i === 0 ? lossCtx.moveTo(x, y) : lossCtx.lineTo(x, y);
  });
  lossCtx.stroke();

  // 渐变填充
  const grad = lossCtx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, 'rgba(79,195,247,0.3)');
  grad.addColorStop(1, 'rgba(79,195,247,0)');
  lossCtx.fillStyle = grad;
  lossCtx.lineTo(w, h);
  lossCtx.lineTo(0, h);
  lossCtx.fill();
}

// ===== 神经网络可视化 =====
const vizCanvas = document.getElementById('vizCanvas');
const vizCtx    = vizCanvas.getContext('2d');

let vizAnimFrame = null;
let vizInputGlow = null;
let vizOutputGlow = null;

function updateNetworkViz(pixels, probs) {
  vizInputGlow  = pixels;
  vizOutputGlow = probs;
  if (vizAnimFrame) cancelAnimationFrame(vizAnimFrame);
  animateViz(0);
}

function animateViz(frame) {
  drawNetworkViz(frame);
  if (frame < 40) {
    vizAnimFrame = requestAnimationFrame(() => animateViz(frame + 1));
  }
}

function drawNetworkViz(frame) {
  const w = vizCanvas.width, h = vizCanvas.height;
  vizCtx.clearRect(0, 0, w, h);

  const t = Math.min(frame / 30, 1); // 动画进度 0→1

  // 动态读取实际网络架构
  const hiddenSizes = nn.hiddenSizes; // 隐藏层各层神经元数
  const hiddenColors = ['#4fc3f7','#81c784','#ce93d8','#ffb74d','#80cbc4','#f48fb1'];

  const layerDefs = [
    { label: `输入层 ${nn.inputSize}`, color: '#ffd54f',
      nodes: sampleNodes(vizInputGlow || new Array(nn.inputSize).fill(0), Math.min(8, nn.inputSize)) },
    ...hiddenSizes.map((sz, i) => ({
      label: `隐层${i+1} ${sz}`,
      color: hiddenColors[i % hiddenColors.length],
      nodes: uniformNodes(Math.min(7, sz))
    })),
    { label: `输出层 ${nn.outputSize}`, color: '#ef9a9a',
      nodes: outputNodes(vizOutputGlow || new Array(labels.length).fill(1/labels.length)) }
  ];

  // 根据层数动态计算 X 坐标
  const totalLayers = layerDefs.length;
  const marginX = 40;
  const layerX = layerDefs.map((_, i) =>
    marginX + (i / (totalLayers - 1)) * (w - marginX * 2)
  );

  // 绘制连线
  for (let l = 0; l < layerDefs.length - 1; l++) {
    const src = layerDefs[l].nodes;
    const dst = layerDefs[l+1].nodes;
    for (let i = 0; i < src.length; i++) {
      for (let j = 0; j < dst.length; j++) {
        const alpha = 0.06 + 0.06 * t;
        vizCtx.strokeStyle = `rgba(255,255,255,${alpha})`;
        vizCtx.lineWidth = 0.5;
        vizCtx.beginPath();
        vizCtx.moveTo(layerX[l], nodeY(h, src.length, i));
        vizCtx.lineTo(layerX[l+1], nodeY(h, dst.length, j));
        vizCtx.stroke();
      }
    }
    // 激活信号动画
    if (t > 0) {
      const sigAlpha = t;
      src.forEach((v, i) => {
        dst.forEach((_, j) => {
          if (Math.random() > 0.85) return;
          const px = layerX[l] + (layerX[l+1] - layerX[l]) * t;
          const py = nodeY(h, src.length, i) + (nodeY(h, dst.length, j) - nodeY(h, src.length, i)) * t;
          vizCtx.beginPath();
          vizCtx.arc(px, py, 2, 0, Math.PI * 2);
          vizCtx.fillStyle = `rgba(255,255,100,${sigAlpha * 0.5})`;
          vizCtx.fill();
        });
      });
    }
  }

  // 绘制节点
  layerDefs.forEach((layer, l) => {
    const x = layerX[l];
    layer.nodes.forEach((val, i) => {
      const y = nodeY(h, layer.nodes.length, i);
      const r = 10;
      const brightness = 0.2 + 0.8 * Math.min(Math.abs(val), 1) * t;
      // 光晕
      const glow = vizCtx.createRadialGradient(x, y, 0, x, y, r * 2);
      glow.addColorStop(0, hexToRgba(layer.color, brightness * 0.6));
      glow.addColorStop(1, 'rgba(0,0,0,0)');
      vizCtx.beginPath();
      vizCtx.arc(x, y, r * 2, 0, Math.PI * 2);
      vizCtx.fillStyle = glow;
      vizCtx.fill();
      // 节点本体
      vizCtx.beginPath();
      vizCtx.arc(x, y, r, 0, Math.PI * 2);
      vizCtx.fillStyle = hexToRgba(layer.color, 0.3 + 0.7 * brightness);
      vizCtx.fill();
      vizCtx.strokeStyle = layer.color;
      vizCtx.lineWidth = 1.5;
      vizCtx.stroke();
      // 输出层显示标签
      if (l === layerDefs.length - 1 && vizOutputGlow) {
        const labelIdx = layer.nodeIndices ? layer.nodeIndices[i] : i;
        vizCtx.fillStyle = '#fff';
        vizCtx.font = '10px sans-serif';
        vizCtx.textAlign = 'left';
        vizCtx.fillText(labels[labelIdx] || labelIdx, x + 14, y + 4);
      }
    });
    // 层标签
    vizCtx.fillStyle = 'rgba(255,255,255,0.5)';
    vizCtx.font = '11px sans-serif';
    vizCtx.textAlign = 'center';
    vizCtx.fillText(layer.label, x, h - 8);
  });
}

function nodeY(h, total, i) {
  const margin = 24;
  const step = (h - margin * 2 - 30) / Math.max(total - 1, 1);
  return margin + i * step;
}

function sampleNodes(arr, n) {
  const step = Math.floor(arr.length / n);
  return Array.from({ length: n }, (_, i) => arr[i * step] || 0);
}

function uniformNodes(n) {
  return Array.from({ length: n }, () => 0.5);
}

function outputNodes(probs) {
  // 取概率最高的前 min(labels.length, 8) 个
  const n = Math.min(labels.length, 8);
  const indexed = probs.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p).slice(0, n);
  const nodes = indexed.map(x => x.p);
  nodes.nodeIndices = indexed.map(x => x.i);
  return nodes;
}

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// ===== 笔刷大小 =====
document.getElementById('brushSize').addEventListener('input', e => {
  ctx.lineWidth = parseInt(e.target.value);
  document.getElementById('brushDisplay').textContent = e.target.value;
});

// ===== Toast 提示 =====
function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2500);
}

// ========================================================
// ===== 网络架构编辑器 =====
// ========================================================

// 当前编辑中的隐藏层配置（神经元数数组）
let editingHiddenSizes = [128, 64];

// 渲染层编辑器 UI
function renderLayerEditor() {
  const container = document.getElementById('layerEditor');
  container.innerHTML = '';

  // 输入层（固定，不可编辑）
  container.appendChild(createLayerRow({
    label: '输入层',
    value: 784,
    color: '#ffd54f',
    editable: false,
    tag: 'IN',
    index: -1
  }));

  // 隐藏层（可编辑、可删除）
  editingHiddenSizes.forEach((size, i) => {
    container.appendChild(createLayerRow({
      label: `隐藏层 ${i + 1}`,
      value: size,
      color: ['#4fc3f7','#81c784','#ce93d8','#ffb74d','#80cbc4','#f48fb1'][i % 6],
      editable: true,
      tag: 'H',
      index: i,
      canDelete: editingHiddenSizes.length > 1
    }));
  });

  // 输出层（显示类别数，不可手动修改）
  container.appendChild(createLayerRow({
    label: '输出层',
    value: labels.length,
    color: '#ef9a9a',
    editable: false,
    tag: 'OUT',
    index: -2
  }));

  // 更新隐藏层数量提示
  const hc = document.getElementById('hiddenCount');
  if (hc) hc.textContent = editingHiddenSizes.length;
}

// 创建单个层行 DOM
function createLayerRow({ label, value, color, editable, tag, index, canDelete }) {
  const row = document.createElement('div');
  row.className = 'layer-editor-row';
  row.style.cssText = `
    display:flex; align-items:center; gap:8px;
    padding:6px 8px; margin-bottom:6px;
    background:var(--bg-panel); border-radius:8px;
    border-left:3px solid ${color};
  `;

  // 标签
  const labelEl = document.createElement('span');
  labelEl.textContent = label;
  labelEl.style.cssText = `font-size:0.78rem;color:var(--text-dim);flex:1;white-space:nowrap;`;
  row.appendChild(labelEl);

  // 激活函数标签
  const actTag = document.createElement('span');
  actTag.textContent = tag === 'OUT' ? 'Softmax' : tag === 'IN' ? 'Input' : 'ReLU';
  actTag.style.cssText = `font-size:0.68rem;color:${color};background:${color}22;padding:2px 6px;border-radius:4px;white-space:nowrap;`;
  row.appendChild(actTag);

  if (editable) {
    // 减号按钮
    const btnMinus = document.createElement('button');
    btnMinus.textContent = '−';
    btnMinus.title = '减少 8 个神经元';
    btnMinus.style.cssText = 'padding:2px 8px;background:rgba(255,255,255,0.08);border:1px solid var(--border);border-radius:4px;color:var(--text);font-size:0.85rem;cursor:pointer;';
    btnMinus.onclick = () => {
      editingHiddenSizes[index] = Math.max(8, editingHiddenSizes[index] - 8);
      renderLayerEditor();
    };
    row.appendChild(btnMinus);

    // 数字输入框
    const input = document.createElement('input');
    input.type = 'number';
    input.value = value;
    input.min = 1;
    input.max = 1024;
    input.style.cssText = `
      width:60px; text-align:center;
      background:var(--bg-card); border:1px solid var(--border);
      border-radius:6px; color:var(--text);
      padding:4px 6px; font-size:0.82rem; font-weight:700;
    `;
    input.oninput = () => {
      const v = parseInt(input.value);
      if (!isNaN(v) && v >= 1 && v <= 1024) {
        editingHiddenSizes[index] = v;
        const hc = document.getElementById('hiddenCount');
        if (hc) hc.textContent = editingHiddenSizes.length;
      }
    };
    input.onblur = () => {
      const v = parseInt(input.value);
      editingHiddenSizes[index] = Math.max(1, Math.min(1024, isNaN(v) ? 32 : v));
      renderLayerEditor();
    };
    row.appendChild(input);

    // 加号按钮
    const btnPlus = document.createElement('button');
    btnPlus.textContent = '＋';
    btnPlus.title = '增加 8 个神经元';
    btnPlus.style.cssText = btnMinus.style.cssText;
    btnPlus.onclick = () => {
      editingHiddenSizes[index] = Math.min(1024, editingHiddenSizes[index] + 8);
      renderLayerEditor();
    };
    row.appendChild(btnPlus);

    // 删除层按钮
    if (canDelete) {
      const btnDel = document.createElement('button');
      btnDel.innerHTML = '🗑';
      btnDel.title = '删除此隐藏层';
      btnDel.style.cssText = 'padding:2px 8px;background:rgba(239,83,80,0.15);border:1px solid rgba(239,83,80,0.3);border-radius:4px;color:#ef5350;cursor:pointer;font-size:0.85rem;';
      btnDel.onclick = () => {
        editingHiddenSizes.splice(index, 1);
        renderLayerEditor();
      };
      row.appendChild(btnDel);
    }
  } else {
    // 不可编辑层只显示神经元数
    const numEl = document.createElement('span');
    numEl.textContent = value;
    numEl.id = tag === 'OUT' ? 'archOutputCount' : '';
    numEl.style.cssText = `font-size:0.85rem;font-weight:700;color:${color};min-width:36px;text-align:center;`;
    row.appendChild(numEl);
  }

  // 上移/下移（仅隐藏层）
  if (editable && editingHiddenSizes.length > 1) {
    const btnUp = document.createElement('button');
    btnUp.textContent = '↑';
    btnUp.title = '向上移动';
    btnUp.style.cssText = 'padding:2px 6px;background:rgba(255,255,255,0.06);border:1px solid var(--border);border-radius:4px;color:var(--text-dim);cursor:pointer;font-size:0.78rem;';
    btnUp.disabled = index === 0;
    btnUp.onclick = () => {
      if (index > 0) {
        [editingHiddenSizes[index-1], editingHiddenSizes[index]] =
          [editingHiddenSizes[index], editingHiddenSizes[index-1]];
        renderLayerEditor();
      }
    };
    row.appendChild(btnUp);

    const btnDown = document.createElement('button');
    btnDown.textContent = '↓';
    btnDown.title = '向下移动';
    btnDown.style.cssText = btnUp.style.cssText;
    btnDown.disabled = index === editingHiddenSizes.length - 1;
    btnDown.onclick = () => {
      if (index < editingHiddenSizes.length - 1) {
        [editingHiddenSizes[index], editingHiddenSizes[index+1]] =
          [editingHiddenSizes[index+1], editingHiddenSizes[index]];
        renderLayerEditor();
      }
    };
    row.appendChild(btnDown);
  }

  return row;
}

// 添加隐藏层
document.getElementById('btnAddLayer').addEventListener('click', () => {
  if (editingHiddenSizes.length >= 6) {
    showToast('最多支持 6 个隐藏层');
    return;
  }
  // 新层默认大小 = 最后一层的一半，最小 16
  const last = editingHiddenSizes[editingHiddenSizes.length - 1] || 64;
  editingHiddenSizes.push(Math.max(16, Math.round(last / 2)));
  renderLayerEditor();
  showToast(`已添加隐藏层 ${editingHiddenSizes.length}（${editingHiddenSizes[editingHiddenSizes.length-1]} 神经元）`);
});

// 应用架构（重建网络）
document.getElementById('btnApplyArch').addEventListener('click', () => {
  // 验证所有隐藏层神经元数
  const valid = editingHiddenSizes.every(s => Number.isInteger(s) && s >= 1 && s <= 1024);
  if (!valid) {
    showToast('请检查神经元数量，每层需在 1~1024 之间');
    return;
  }
  if (!confirm(
    `确定应用新架构？\n\n结构：784 → ${editingHiddenSizes.join(' → ')} → ${labels.length}\n\n⚠️ 训练权重将被清除！`
  )) return;

  stopTraining();
  nn = new NeuralNetwork(784, [...editingHiddenSizes], labels.length);
  trainingHistory = [];
  totalEpoch = 0;

  updateLossChart();
  document.getElementById('epochCount').textContent = '迭代：0';
  document.getElementById('lossDisplay').textContent = '损失：-';
  document.getElementById('resultLabel').textContent = '?';
  document.getElementById('probBars').innerHTML = '';

  // 更新 header 描述
  updateHeaderDesc();
  // 更新输出层信息
  const oi = document.getElementById('outputInfo');
  if (oi) oi.textContent = labels.length + ' 类别';
  // 刷新可视化
  drawNetworkViz(0);

  showToast(`✅ 架构已应用：784 → ${editingHiddenSizes.join(' → ')} → ${labels.length}`);
});

// 更新 header 副标题
function updateHeaderDesc() {
  const el = document.getElementById('headerDesc');
  if (!el) return;
  el.textContent = `784 → ${nn.hiddenSizes.join(' → ')} → ${nn.outputSize} · Softmax · 支持自定义符号训练`;
}

// ===== 初始化 =====
function init() {
  initCanvas();
  updateLabelList();
  renderLayerEditor();
  updateHeaderDesc();
  drawNetworkViz(0);
  updateLossChart();
}

window.addEventListener('load', init);
