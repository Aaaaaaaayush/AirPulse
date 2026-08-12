/**
 * AirPulse — Dashboard Frontend Application.
 * Handles real-time API communication, state management, and Chart.js forecast rendering.
 */

document.addEventListener('DOMContentLoaded', () => {
  let currentCity = 'mumbai';
  let chartInstance = null;

  const cityPills = document.querySelectorAll('.city-pill');
  const cityNameEl = document.getElementById('cityName');
  const currentAqiEl = document.getElementById('currentAqi');
  const aqiBadgeEl = document.getElementById('aqiBadge');
  const tempValEl = document.getElementById('tempVal');
  const humidityValEl = document.getElementById('humidityVal');
  const windValEl = document.getElementById('windVal');
  const precipValEl = document.getElementById('precipVal');
  const lastUpdatedEl = document.getElementById('lastUpdated');
  const modelVerEl = document.getElementById('modelVer');
  const modelStageEl = document.getElementById('modelStage');

  // Fetch Health Metadata
  async function fetchHealth() {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        const data = await res.json();
        if (modelVerEl) modelVerEl.textContent = `v${data.model_version}`;
        if (modelStageEl) modelStageEl.textContent = data.model_stage;
      }
    } catch (err) {
      console.warn('Could not fetch health metadata:', err);
    }
  }

  // Fetch City Forecast
  async function fetchForecast(city) {
    try {
      currentAqiEl.textContent = '...';
      const res = await fetch(`/api/forecast?city=${city}`);
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      const data = await res.json();

      updateUI(data);
    } catch (err) {
      console.error(`Failed to fetch forecast for ${city}:`, err);
      currentAqiEl.textContent = 'ERR';
    }
  }

  // Update UI Elements
  function updateUI(data) {
    cityNameEl.textContent = data.city_display;
    currentAqiEl.textContent = Math.round(data.current_aqi);
    currentAqiEl.style.color = data.current_color;

    aqiBadgeEl.textContent = data.current_category;
    aqiBadgeEl.style.backgroundColor = data.current_color;

    if (data.forecast && data.forecast.length > 0) {
      const latest = data.forecast[0];
      tempValEl.textContent = `${latest.temperature_2m} °C`;
      humidityValEl.textContent = `${latest.relative_humidity_2m} %`;
      windValEl.textContent = `${latest.wind_speed_10m} km/h`;
      precipValEl.textContent = `${latest.precipitation} mm`;
    }

    if (lastUpdatedEl) {
      const d = new Date(data.fetched_at);
      lastUpdatedEl.textContent = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    renderChart(data.forecast);
  }

  // Render Chart.js Forecast Curve
  function renderChart(forecastList) {
    const ctx = document.getElementById('forecastChart').getContext('2d');

    const labels = forecastList.map(p => {
      const date = new Date(p.timestamp);
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    });

    const aqiValues = forecastList.map(p => p.aqi);
    const pointColors = forecastList.map(p => p.color);

    if (chartInstance) {
      chartInstance.destroy();
    }

    // Gradient Fill
    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, 'rgba(59, 130, 246, 0.4)');
    gradient.addColorStop(1, 'rgba(59, 130, 246, 0.0)');

    chartInstance = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: '24-Hour Forecasted AQI',
          data: aqiValues,
          borderColor: '#3B82F6',
          borderWidth: 3,
          fill: true,
          backgroundColor: gradient,
          tension: 0.35,
          pointBackgroundColor: pointColors,
          pointBorderColor: '#FFFFFF',
          pointBorderWidth: 2,
          pointRadius: 5,
          pointHoverRadius: 8,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(18, 24, 38, 0.9)',
            titleFont: { family: 'Outfit', size: 14, weight: 'bold' },
            bodyFont: { family: 'Inter', size: 13 },
            borderColor: 'rgba(255, 255, 255, 0.1)',
            borderWidth: 1,
            padding: 12,
            callbacks: {
              label: function(context) {
                const p = forecastList[context.dataIndex];
                return ` Predicted AQI: ${p.aqi} (${p.category})`;
              }
            }
          }
        },
        scales: {
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94A3B8', font: { family: 'Inter', size: 11 } }
          },
          y: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94A3B8', font: { family: 'Inter', size: 11 } },
            beginAtZero: false
          }
        }
      }
    });
  }

  // Event Listeners for City Switcher
  cityPills.forEach(pill => {
    pill.addEventListener('click', () => {
      cityPills.forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      currentCity = pill.dataset.city;
      fetchForecast(currentCity);
    });
  });

  // Initial Load
  fetchHealth();
  fetchForecast(currentCity);
});
