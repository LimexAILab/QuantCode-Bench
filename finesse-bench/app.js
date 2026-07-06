const CHART_COLORS = [
  '#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
  '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1',
];

const state = {
  leaderboardMetric: 'public',
  compareMetric: 'public',
  leaderboardSearch: '',
  leaderboardExpanded: false,
  selectedModels: [
    'GPT-5.5',
    'Claude Opus 4.8',
    'Claude Sonnet 4.6',
    'Kimi K2.5',
    'GLM-5.2',
  ],
  showCI: true,
};

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

let compareChartInstance = null;

const METRIC_ALIASES = {
  public: 'public_benchs',
  exam: 'exam_like',
  trading: 'ta_benchs',
};

const CI_LOOKUP = Array.isArray(CI_DATA)
  ? Object.assign({}, ...CI_DATA)
  : CI_DATA;

function readMetric(metrics, metric) {
  if (!metrics) return null;
  if (metrics[metric] != null) return metrics[metric];

  const alias = METRIC_ALIASES[metric];
  if (alias && metrics[alias] != null) return metrics[alias];

  return null;
}

function pct(value) {
  return value == null ? '—' : `${(value * 100).toFixed(1)}%`;
}

function delta(a, b) {
  if (a == null || b == null) return null;
  return a - b;
}



Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
Chart.defaults.font.size = 12;


const errorBarPlugin = {
  id: 'errorBars',
  afterDatasetsDraw(chart, args, pluginOptions) {
    if (!pluginOptions?.enabled || !pluginOptions?.errorBars?.length) return;
    const { ctx, scales } = chart;
    const meta = chart.getDatasetMeta(0);
    ctx.save();
    ctx.strokeStyle = '#111827';
    ctx.lineWidth = 1.25;

    meta.data.forEach((bar, i) => {
      const error = pluginOptions.errorBars[i];
      if (!error) return;
      const x = bar.x;
      const yTop = scales.y.getPixelForValue(error.high);
      const yBottom = scales.y.getPixelForValue(error.low);

      ctx.beginPath();
      ctx.moveTo(x, yTop);
      ctx.lineTo(x, yBottom);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(x - 5, yTop);
      ctx.lineTo(x + 5, yTop);
      ctx.moveTo(x - 5, yBottom);
      ctx.lineTo(x + 5, yBottom);
      ctx.stroke();
    });

    ctx.restore();
  },
};

Chart.register(errorBarPlugin);

function metricLabel(key) {
  const found = ALL_METRICS.find(m => m.key === key);
  return found ? found.label : key;
}

function getModel(modelName) {
  return MODEL_DATA.find(m => m.model === modelName);
}

function getMetricValue(modelName, metric) {
  const model = getModel(modelName);
  return readMetric(model?.metrics, metric);
}

function getCI(modelName, metric) {
  const modelCI = CI_LOOKUP?.[modelName];
  if (!modelCI) return null;
  if (modelCI[metric] != null) return modelCI[metric];
  const alias = METRIC_ALIASES[metric];
  return alias ? (modelCI[alias] ?? null) : null;
}

function mean(arr) {
  if (!arr.length) return null;
  return arr.reduce((s, x) => s + x, 0) / arr.length;
}

function std(arr) {
  if (!arr.length) return null;
  const m = mean(arr);
  return Math.sqrt(mean(arr.map(x => (x - m) ** 2)));
}

function topRows(metric, count = 3) {
  return [...MODEL_DATA]
    .map(row => ({ ...row, _score: getMetricValue(row.model, metric) }))
    .filter(row => row._score != null)
    .sort((a, b) => b._score - a._score)
    .slice(0, count);
}


function getTop5Overall() {
  return [...MODEL_DATA]
    .map(row => {
      const values = ['public', 'exam', 'trading']
        .map(k => getMetricValue(row.model, k))
        .filter(v => v != null);
      return { model: row.model, score: mean(values) };
    })
    .filter(row => row.score != null)
    .sort((a, b) => b.score - a.score)
    .slice(0, 5)
    .map(x => x.model);
}


function populateSelect(selectId, value) {
  const select = document.getElementById(selectId);
  select.innerHTML = ALL_METRICS.map(
    m => `<option value="${m.key}" ${m.key === value ? 'selected' : ''}>${m.label}</option>`
  ).join('');
}

function copyBibtex() {
  const code = document.querySelector('.bibtex-block code');
  navigator.clipboard.writeText(code.textContent).then(() => {
    const btn = document.querySelector('.bibtex-block + button, .bibtex-block ~ button');
    if (!btn) return;
    const orig = btn.textContent;
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = orig; }, 1500);
  });
}

function renderGroupCards() {
  const container = document.getElementById('groupCards');
  container.innerHTML = GROUP_METRICS.map(group => {
    const top = topRows(group.key, 3);
    return `
      <article class="group-card">
        <h3>${group.label}</h3>
        <p class="mt-2">${GROUP_DESCRIPTIONS[group.key]}</p>
        <div class="mt-4">
          ${top.map((row, i) => `
            <div class="mini-rank">
              <span class="${i === 0 ? 'rank-gold' : i === 1 ? 'rank-silver' : 'rank-bronze'}">${i + 1}</span>
              <span>${row.model}</span>
              <span class="mini-rank-score">${pct(row._score)}</span>
            </div>
          `).join('')}
        </div>
        <div class="mt-4">
          <button class="btn-secondary" onclick="jumpToMetric('${group.key}')">View filtered leaderboard</button>
        </div>
      </article>
    `;
  }).join('');
}

function jumpToMetric(metric) {
  state.leaderboardMetric = metric;
  document.getElementById('leaderboardMetric').value = metric;
  renderLeaderboard();
  document.getElementById('leaderboard').scrollIntoView({ behavior: 'smooth' });
}

function renderBalancedModels() {
  const container = document.getElementById('balancedModels');
  const rows = MODEL_DATA.map(row => {
    const vals = ['public', 'exam', 'trading']
      .map(k => getMetricValue(row.model, k))
      .filter(v => v != null);

    return {
      model: row.model,
      mean: mean(vals),
      min: vals.length ? Math.min(...vals) : null,
      spread: std(vals),
    };
  })
    .filter(row => row.mean != null)
    .sort((a, b) => b.mean - a.mean)
    .slice(0, 5);

  container.innerHTML = rows.map(row => `
    <div class="balanced-item">
      <div class="balanced-item-title">${row.model}</div>
      <div class="balanced-metrics">
        <div><span>mean</span><strong>${pct(row.mean)}</strong></div>
        <div><span>min</span><strong>${pct(row.min)}</strong></div>
        <div><span>std</span><strong>${row.spread == null ? '—' : (row.spread * 100).toFixed(2)}</strong></div>
      </div>
    </div>
  `).join('');
}


function renderLeaderboard() {
  const tbody = document.getElementById('leaderboardBody');
  const toggleBtn = document.getElementById('leaderboardToggle');
  const metric = state.leaderboardMetric;
  const query = state.leaderboardSearch.trim().toLowerCase();

  let rows = MODEL_DATA
    .map(row => ({
      ...row,
      _score: getMetricValue(row.model, metric),
      _public: getMetricValue(row.model, 'public'),
      _exam: getMetricValue(row.model, 'exam'),
      _trading: getMetricValue(row.model, 'trading'),
    }))
    .filter(row => row._score != null)
    .filter(row => row.model.toLowerCase().includes(query))
    .sort((a, b) => b._score - a._score);

  const visibleRows = state.leaderboardExpanded ? rows : rows.slice(0, 10);

  tbody.innerHTML = visibleRows.map((row, idx) => {
    const rankClass =
      idx === 0 ? 'rank-gold' :
      idx === 1 ? 'rank-silver' :
      idx === 2 ? 'rank-bronze' : '';

    const ci = getCI(row.model, metric);
    const ciText = ci ? `${pct(ci.low)} – ${pct(ci.high)}` : '—';

    const dPublicExam = delta(row._public, row._exam);
    const dPublicTrading = delta(row._public, row._trading);

    return `
      <tr>
        <td class="${rankClass}">${idx + 1}</td>
        <td>${row.model}</td>
        <td class="font-semibold">${pct(row._score)}</td>
        <td>${ciText}</td>
        <td>${dPublicExam == null ? '—' : (dPublicExam * 100).toFixed(2)}</td>
        <td>${dPublicTrading == null ? '—' : (dPublicTrading * 100).toFixed(2)}</td>
      </tr>
    `;
  }).join('');

  if (rows.length > 10) {
    toggleBtn.classList.remove('hidden');
    toggleBtn.textContent = state.leaderboardExpanded ? 'Show top 10' : 'Show all';
  } else {
    toggleBtn.classList.add('hidden');
  }
}



function renderModelPicker() {
  const container = document.getElementById('modelPicker');
  const rows = [...MODEL_DATA].sort(
    (a, b) => (getMetricValue(b.model, 'public') ?? -1) - (getMetricValue(a.model, 'public') ?? -1)
  );

  container.innerHTML = rows.map(row => {
    const checked = state.selectedModels.includes(row.model);
    return `
      <label class="model-option">
        <input type="checkbox" value="${row.model}" ${checked ? 'checked' : ''} />
        <div class="flex-1 min-w-0">
          <div class="text-sm text-gray-900 truncate">${row.model}</div>
          <div class="meta">
            public ${pct(getMetricValue(row.model, 'public'))}
            · exam ${pct(getMetricValue(row.model, 'exam'))}
            · TA ${pct(getMetricValue(row.model, 'trading'))}
          </div>
        </div>
      </label>
    `;
  }).join('');

  container.querySelectorAll('input[type="checkbox"]').forEach(input => {
    input.addEventListener('change', e => {
      const model = e.target.value;
      if (e.target.checked) {
        if (state.selectedModels.length >= 10) {
          e.target.checked = false;
          alert('You can compare up to 10 models.');
          return;
        }
        state.selectedModels.push(model);
      } else {
        state.selectedModels = state.selectedModels.filter(x => x !== model);
      }
      updateSelectedCount();
      renderCompareChart();
    });
  });

  updateSelectedCount();
}


function updateSelectedCount() {
  document.getElementById('selectedCount').textContent = `${state.selectedModels.length} / 10`;
}

function renderCompareChart() {
  const metric = state.compareMetric;
  const rows = state.selectedModels
    .map(model => ({
      model,
      score: getMetricValue(model, metric),
      ci: getCI(model, metric),
    }))
    .filter(row => row.score != null)
    .sort((a, b) => b.score - a.score);

  const showCI = state.showCI;
  const errorBars = rows.map(row => row.ci ? { low: row.ci.low, high: row.ci.high } : null);
  const hasAnyCI = errorBars.some(Boolean);

  const ciNote = document.getElementById('ciNote');
  ciNote.classList.toggle('hidden', !(showCI && !hasAnyCI));

  let yMin = 0;
  let yMax = 1;

  if (rows.length) {
    const lows = rows.map(row => (showCI && row.ci ? row.ci.low : row.score));
    const highs = rows.map(row => (showCI && row.ci ? row.ci.high : row.score));

    yMin = clamp(Math.min(...lows) - 0.03, 0, 1);
    yMax = clamp(Math.max(...highs) + 0.03, 0, 1);

    if (yMax <= yMin) {
      yMin = clamp(yMin - 0.01, 0, 1);
      yMax = clamp(yMax + 0.01, 0, 1);
    }
  }

  if (compareChartInstance) compareChartInstance.destroy();

  compareChartInstance = new Chart(document.getElementById('compareChart'), {
    type: 'bar',
    data: {
      labels: rows.map(r => r.model),
      datasets: [{
        label: metricLabel(metric),
        data: rows.map(r => r.score),
        backgroundColor: rows.map((_, i) => CHART_COLORS[i % CHART_COLORS.length] + 'CC'),
        borderColor: rows.map((_, i) => CHART_COLORS[i % CHART_COLORS.length]),
        borderWidth: 1,
        borderRadius: 6,
        borderSkipped: false,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        errorBars: {
          enabled: showCI && hasAnyCI,
          errorBars,
        },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const row = rows[ctx.dataIndex];
              if (showCI && row.ci) {
                return `${pct(row.score)} (CI ${pct(row.ci.low)} – ${pct(row.ci.high)})`;
              }
              return pct(row.score);
            },
          },
        },
      },
      scales: {
        y: {
          min: yMin,
          max: yMax,
          grid: { color: '#f3f4f6' },
          ticks: {
            callback: value => `${Math.round(value * 100)}%`,
          },
        },
        x: {
          grid: { display: false },
          ticks: { maxRotation: 40, minRotation: 20 },
        },
      },
    },
  });
}



function renderDatasetExplorer() {
  const container = document.getElementById('datasetExplorer');
  container.innerHTML = DATASET_META.map((d, i) => `
    <div class="accordion ${i === 0 ? 'open' : ''}">
      <button class="accordion-header" onclick="toggleAccordion(this)">
        <span>${d.name}</span>
        <svg class="accordion-chevron w-5 h-5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path d="M19 9l-7 7-7-7"/>
        </svg>
      </button>
      <div class="accordion-body">
        <div class="dataset-badges">
          <span class="dataset-badge">${d.count} questions</span>
          <span class="dataset-badge gray">${d.format}</span>
          <span class="dataset-badge gray">${d.language}</span>
        </div>
        <div class="dataset-body">${d.description}</div>
      </div>
    </div>
  `).join('');
}

function toggleAccordion(btn) {
  const acc = btn.closest('.accordion');
  acc.classList.toggle('open');
}

function bindControls() {
  populateSelect('leaderboardMetric', state.leaderboardMetric);
  populateSelect('compareMetric', state.compareMetric);

  document.getElementById('leaderboardMetric').addEventListener('change', e => {
    state.leaderboardMetric = e.target.value;
    state.leaderboardExpanded = false;
    renderLeaderboard();
  });

  document.getElementById('leaderboardSearch').addEventListener('input', e => {
    state.leaderboardSearch = e.target.value;
    state.leaderboardExpanded = false;
    renderLeaderboard();
  });

  document.getElementById('leaderboardReset').addEventListener('click', () => {
    state.leaderboardMetric = 'public';
    state.leaderboardSearch = '';
    state.leaderboardExpanded = false;
    document.getElementById('leaderboardMetric').value = 'public';
    document.getElementById('leaderboardSearch').value = '';
    renderLeaderboard();
  });


  document.getElementById('compareMetric').addEventListener('change', e => {
    state.compareMetric = e.target.value;
    renderCompareChart();
  });

  document.getElementById('compareCI').addEventListener('change', e => {
    state.showCI = e.target.checked;
    renderCompareChart();
  });

  document.getElementById('compareTop5').addEventListener('click', () => {
    state.selectedModels = getTop5Overall();
    renderModelPicker();
    renderCompareChart();
  });

  document.getElementById('compareReset').addEventListener('click', () => {
    state.selectedModels = [];
    renderModelPicker();
    renderCompareChart();
  });
  
  document.getElementById('leaderboardToggle').addEventListener('click', () => {
    state.leaderboardExpanded = !state.leaderboardExpanded;
    renderLeaderboard();
  });


}

document.addEventListener('DOMContentLoaded', () => {
  renderGroupCards();
  renderBalancedModels();
  renderDatasetExplorer();
  bindControls();
  renderModelPicker();
  renderLeaderboard();
  renderCompareChart();
});

