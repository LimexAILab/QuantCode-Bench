/* ===== Benchmark Tab Switching ===== */
const benchmarks = [
  { id: 'quantcode-bench', name: 'QuantCode-Bench', active: true },
];

document.querySelectorAll('.bench-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    const benchId = tab.dataset.bench;
    document.querySelectorAll('.bench-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.bench-content').forEach(c => c.classList.remove('active'));
    tab.classList.add('active');
    const target = document.getElementById('bench-' + benchId);
    if (target) target.classList.add('active');
  });
});

/* ===== Accordion ===== */
function toggleAccordion(btn) {
  const acc = btn.closest('.accordion');
  acc.classList.toggle('open');
}

/* ===== Copy BibTeX ===== */
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

/* ===== Table Sorting ===== */
function initSortableTables() {
  document.querySelectorAll('.sortable').forEach(table => {
    const headers = table.querySelectorAll('.sortable-col');
    let currentSort = { col: 4, dir: 'desc' };

    function sortTable(colIdx, dir) {
      const tbody = table.querySelector('tbody');
      const rows = Array.from(tbody.querySelectorAll('tr'));
      rows.sort((a, b) => {
        const aVal = a.children[colIdx + 1]?.textContent.trim() || '';
        const bVal = b.children[colIdx + 1]?.textContent.trim() || '';
        const aNum = parseFloat(aVal);
        const bNum = parseFloat(bVal);
        if (!isNaN(aNum) && !isNaN(bNum)) {
          return dir === 'asc' ? aNum - bNum : bNum - aNum;
        }
        return dir === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
      });
      rows.forEach((row, i) => {
        row.children[0].textContent = i + 1;
        row.children[0].className = '';
        if (i === 0) row.children[0].className = 'rank-gold';
        else if (i === 1) row.children[0].className = 'rank-silver';
        else if (i === 2) row.children[0].className = 'rank-bronze';
        tbody.appendChild(row);
      });
    }

    headers.forEach(th => {
      th.addEventListener('click', () => {
        const col = parseInt(th.dataset.col);
        let dir = 'desc';
        if (currentSort.col === col && currentSort.dir === 'desc') dir = 'asc';
        currentSort = { col, dir };

        headers.forEach(h => {
          const ind = h.querySelector('.sort-indicator');
          if (ind) ind.remove();
        });
        const indicator = document.createElement('span');
        indicator.className = 'sort-indicator';
        indicator.innerHTML = dir === 'desc' ? ' &#9660;' : ' &#9650;';
        th.appendChild(indicator);

        sortTable(col, dir);
      });
    });

    sortTable(currentSort.col, currentSort.dir);
  });
}

/* ===== Charts ===== */
const CHART_COLORS = [
  '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
  '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1',
  '#14b8a6', '#e11d48'
];

function initCharts() {
  Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
  Chart.defaults.font.size = 12;

  // Source distribution donut
  new Chart(document.getElementById('chartSources'), {
    type: 'doughnut',
    data: {
      labels: ['Reddit', 'TradingView', 'StackExchange', 'GitHub', 'Synthetic'],
      datasets: [{
        data: [183, 100, 90, 19, 8],
        backgroundColor: ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'],
        borderWidth: 2,
        borderColor: '#fff',
      }]
    },
    options: {
      cutout: '55%',
      plugins: {
        legend: { position: 'bottom', labels: { padding: 16 } },
      }
    }
  });

  // Single-turn Judge Pass bar chart
  const singleData = [
    { model: 'claude-opus-4.6', judge: 75.8 },
    { model: 'gpt-5.4', judge: 70.2 },
    { model: 'claude-sonnet-4.5', judge: 69.8 },
    { model: 'gpt-5.2-codex', judge: 67.5 },
    { model: 'glm-5', judge: 65.4 },
    { model: 'claude-sonnet-4.6', judge: 65.0 },
    { model: 'kimi-k2.5', judge: 64.8 },
    { model: 'gemini-3-flash', judge: 59.8 },
    { model: 'grok-4.1-fast', judge: 48.9 },
    { model: 'deepseek-v3.2', judge: 48.8 },
    { model: 'qwen3-235b', judge: 48.2 },
    { model: 'qwen3-coder-30b', judge: 39.2 },
    { model: 'gemini-2.5-flash', judge: 31.2 },
    { model: 'qwen3-14b', judge: 25.2 },
    { model: 'qwen3-8b', judge: 18.5 },
    { model: 'qwen3-4b', judge: 12.3 },
    { model: 'qwen3-1.7b', judge: 7.8 },
  ];

  new Chart(document.getElementById('chartSingleJudge'), {
    type: 'bar',
    data: {
      labels: singleData.map(d => d.model),
      datasets: [{
        label: 'Judge Pass %',
        data: singleData.map(d => d.judge),
        backgroundColor: singleData.map((_, i) => {
          const t = i / (singleData.length - 1);
          return `rgba(37, 99, 235, ${1 - t * 0.7})`;
        }),
        borderRadius: 4,
        borderSkipped: false,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { color: '#f3f4f6' }, max: 85 },
        y: { grid: { display: false } }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => `Judge Pass: ${ctx.parsed.x}%`
          }
        }
      }
    }
  });

  // Agentic cumulative turns line chart (top 6 models)
  const topModels = [
    { name: 'claude-opus-4.6', turns: [75.8, null, 95.2, null, 97.5, null, null, null, null, 97.5] },
    { name: 'claude-sonnet-4.6', turns: [65.0, null, 90.2, null, 93.8, null, null, null, null, 96.0] },
    { name: 'gpt-5.4', turns: [70.2, null, 91.5, null, 93.2, null, null, null, null, 95.0] },
    { name: 'kimi-k2.5', turns: [64.8, null, 84.2, null, 89.2, null, null, null, null, 93.5] },
    { name: 'claude-sonnet-4.5', turns: [69.8, null, 90.0, null, 91.2, null, null, null, null, 93.0] },
    { name: 'gemini-3-flash', turns: [59.8, null, 83.5, null, 88.2, null, null, null, null, 91.8] },
  ];

  const turnLabels = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10'];

  new Chart(document.getElementById('chartAgenticTurns'), {
    type: 'line',
    data: {
      labels: turnLabels,
      datasets: topModels.map((m, i) => ({
        label: m.name,
        data: m.turns,
        borderColor: CHART_COLORS[i],
        backgroundColor: CHART_COLORS[i] + '20',
        borderWidth: 2.5,
        pointRadius: 4,
        pointHoverRadius: 6,
        tension: 0.3,
        spanGaps: true,
      }))
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 50, max: 100,
          grid: { color: '#f3f4f6' },
          title: { display: true, text: 'Judge Pass %' }
        },
        x: {
          grid: { display: false },
          title: { display: true, text: 'Turn' }
        }
      },
      plugins: {
        legend: { position: 'bottom', labels: { padding: 16 } },
        tooltip: {
          callbacks: {
            label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y}%`
          }
        }
      }
    }
  });

}

/* ===== Init ===== */
document.addEventListener('DOMContentLoaded', () => {
  initSortableTables();
  initCharts();
});
