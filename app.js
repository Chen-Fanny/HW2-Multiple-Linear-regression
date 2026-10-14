/**
 * CityBike AI Analytics - Interactive Dashboard Logic
 * Implementation of Real-time Multiple Linear Regression Inference & 95% Conformal Prediction
 */

// Model Weights and Parameters from models/best_linear_regression.joblib
const MODEL_CONFIG = {
    intercept: -6.573953,
    conformal_q: 318.81,
    coefficients: {
        is_rush_hour: 164.9184,
        hr_sin: -62.8643,
        hr_cos: -55.5026,
        atemp: 65.2820,
        hr_cos2: -26.5782,
        weather_bad: -56.5501,
        temp: 280.0838,
        yr: 70.7463,
        temp_hum: -223.3540,
        season_winter: 57.2898,
        season_summer: 38.6391,
        season_fall: 30.7861,
        hr_sin2: -18.2944,
        workingday: -25.4453,
        hum: 50.1108,
        windspeed: -32.0992,
        weather_mist: -8.5909,
        holiday: -16.5172
    }
};

document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initCrispTabs();
    initSimulator();
    initScrollSpy();
});

/* -------------------------------------------------------------------------
   1. Theme Toggle (Dark / Light)
   ------------------------------------------------------------------------- */
function initTheme() {
    const toggleBtn = document.getElementById('theme-toggle');
    const savedTheme = localStorage.getItem('theme') || 'dark';

    if (savedTheme === 'light') {
        document.body.classList.remove('dark-theme');
        document.body.classList.add('light-theme');
        toggleBtn.innerHTML = '<i class="fa-solid fa-sun"></i>';
    } else {
        document.body.classList.add('dark-theme');
        document.body.classList.remove('light-theme');
        toggleBtn.innerHTML = '<i class="fa-solid fa-moon"></i>';
    }

    toggleBtn.addEventListener('click', () => {
        if (document.body.classList.contains('dark-theme')) {
            document.body.classList.remove('dark-theme');
            document.body.classList.add('light-theme');
            toggleBtn.innerHTML = '<i class="fa-solid fa-sun"></i>';
            localStorage.setItem('theme', 'light');
        } else {
            document.body.classList.add('dark-theme');
            document.body.classList.remove('light-theme');
            toggleBtn.innerHTML = '<i class="fa-solid fa-moon"></i>';
            localStorage.setItem('theme', 'dark');
        }
    });
}

/* -------------------------------------------------------------------------
   2. CRISP-DM Tabs Switching
   ------------------------------------------------------------------------- */
function initCrispTabs() {
    const tabs = document.querySelectorAll('.crisp-tab');
    const panels = document.querySelectorAll('.crisp-panel');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const step = tab.getAttribute('data-step');
            
            // Deactivate all
            tabs.forEach(t => t.classList.remove('active'));
            panels.forEach(p => p.classList.remove('active'));

            // Activate current
            tab.classList.add('active');
            const targetPanel = document.getElementById(`crisp-panel-${step}`);
            if (targetPanel) {
                targetPanel.classList.add('active');
            }
        });
    });
}

/* -------------------------------------------------------------------------
   3. Real-time Prediction Simulator
   ------------------------------------------------------------------------- */
function initSimulator() {
    // Elements
    const sHour = document.getElementById('sim-hour');
    const sTemp = document.getElementById('sim-temp');
    const sHum = document.getElementById('sim-hum');
    const sWind = document.getElementById('sim-wind');
    const sWorkingday = document.getElementById('sim-workingday');
    const sSeason = document.getElementById('sim-season');
    const sWeather = document.getElementById('sim-weather');
    const sYear = document.getElementById('sim-year');

    // Badges
    const bHour = document.getElementById('val-hour');
    const bTemp = document.getElementById('val-temp');
    const bHum = document.getElementById('val-hum');
    const bWind = document.getElementById('val-wind');

    // Outputs
    const outPred = document.getElementById('pred-output-val');
    const outLower = document.getElementById('pred-lower-val');
    const outUpper = document.getElementById('pred-upper-val');
    const alertBox = document.getElementById('dispatch-alert');
    const alertTitle = document.getElementById('alert-title');
    const alertDesc = document.getElementById('alert-desc');
    const metaRush = document.getElementById('meta-rush-val');

    function updateSimulation() {
        const hour = parseInt(sHour.value, 10);
        const tempC = parseFloat(sTemp.value);
        const humPct = parseFloat(sHum.value);
        const windKm = parseFloat(sWind.value);
        const workingday = parseInt(sWorkingday.value, 10);
        const season = parseInt(sSeason.value, 10);
        const weather = parseInt(sWeather.value, 10);
        const yr = parseInt(sYear.value, 10);

        // Update badge text
        bHour.textContent = `${String(hour).padStart(2, '0')}:00`;
        bTemp.textContent = `${tempC.toFixed(1)} °C`;
        bHum.textContent = `${Math.round(humPct)} %`;
        bWind.textContent = `${Math.round(windKm)} km/h`;

        // Normalize features according to UCI Bike Sharing specification
        // Temp: (t - t_min) / (t_max - t_min), t_min=-8, t_max=39
        const normTemp = Math.max(0, Math.min(1, (tempC - (-8)) / (39 - (-8))));
        
        // Approximate feeling temperature (atemp) based on wind chill / heat index
        const atempC = tempC + (humPct > 60 ? (humPct - 60) * 0.05 : 0) - (windKm * 0.1);
        const normAtemp = Math.max(0, Math.min(1, (atempC - (-16)) / (50 - (-16))));
        
        // Humidity: hum / 100
        const normHum = Math.max(0, Math.min(1, humPct / 100.0));
        
        // Wind speed: wind / 67
        const normWind = Math.max(0, Math.min(1, windKm / 67.0));

        // Rush hour indicator
        const isRushHour = (workingday === 1 && [8, 9, 17, 18].includes(hour)) ? 1 : 0;
        if (isRushHour) {
            metaRush.textContent = '啟動 (貢獻 +164.9 輛)';
            metaRush.className = 'text-success';
        } else {
            metaRush.textContent = '未觸發 (非工作日尖峰)';
            metaRush.className = 'text-muted';
        }

        // Cyclical Harmonics
        const hrRad = (2.0 * Math.PI * hour) / 24.0;
        const hrSin = Math.sin(hrRad);
        const hrCos = Math.cos(hrRad);
        const hrSin2 = Math.sin(2.0 * hrRad);
        const hrCos2 = Math.cos(2.0 * hrRad);

        // Dummies
        const holiday = (workingday === 0 && (hour === 12)) ? 0 : 0; // standard baseline
        const seasonSummer = season === 2 ? 1 : 0;
        const seasonFall = season === 3 ? 1 : 0;
        const seasonWinter = season === 4 ? 1 : 0;
        const weatherMist = weather === 2 ? 1 : 0;
        const weatherBad = weather >= 3 ? 1 : 0;
        const tempHum = normTemp * normHum;

        // Linear Dot Product
        const C = MODEL_CONFIG.coefficients;
        let yHat = MODEL_CONFIG.intercept;
        yHat += C.is_rush_hour * isRushHour;
        yHat += C.hr_sin * hrSin;
        yHat += C.hr_cos * hrCos;
        yHat += C.atemp * normAtemp;
        yHat += C.hr_cos2 * hrCos2;
        yHat += C.weather_bad * weatherBad;
        yHat += C.temp * normTemp;
        yHat += C.yr * yr;
        yHat += C.temp_hum * tempHum;
        yHat += C.season_winter * seasonWinter;
        yHat += C.season_summer * seasonSummer;
        yHat += C.season_fall * seasonFall;
        yHat += C.hr_sin2 * hrSin2;
        yHat += C.workingday * workingday;
        yHat += C.hum * normHum;
        yHat += C.windspeed * normWind;
        yHat += C.weather_mist * weatherMist;
        yHat += C.holiday * holiday;

        // Truncate non-negative
        const predDemand = Math.max(0, Math.round(yHat));
        const q = MODEL_CONFIG.conformal_q;
        const lowerBound = Math.max(0, Math.round(predDemand - q));
        const upperBound = Math.round(predDemand + q);

        // Render Outputs
        outPred.textContent = predDemand;
        outLower.textContent = `${lowerBound} 輛`;
        outUpper.textContent = `${upperBound} 輛`;

        // Update alert level
        alertBox.className = 'alert-box';
        if (predDemand >= 380) {
            alertBox.classList.add('alert-high');
            alertTitle.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> 尖峰超載預警 (High Demand Alert)';
            alertDesc.textContent = `當前時段借車預測高達 ${predDemand} 輛 (95% 預測上限: ${upperBound} 輛)！建議立即指派 2-3 輛大容量卡車對交通樞紐與辦公熱區提早補滿車架！`;
        } else if (predDemand >= 150) {
            alertBox.classList.add('alert-med');
            alertTitle.innerHTML = '<i class="fa-solid fa-circle-exclamation"></i> 常態營運狀態 (Normal Demand)';
            alertDesc.textContent = `預測需求為 ${predDemand} 輛，處於穩定週轉範圍。請保持巡檢調度，監控熱門還車站點車位充足度。`;
        } else {
            alertBox.classList.add('alert-low');
            alertTitle.innerHTML = '<i class="fa-solid fa-circle-check"></i> 離峰平穩時段 (Low Demand Period)';
            alertDesc.textContent = `預測需求僅 ${predDemand} 輛，處於夜間或極端天候離峰。適宜進行車體安全維修檢測、胎壓充填與車載智慧鎖保養。`;
        }
    }

    // Attach listeners
    [sHour, sTemp, sHum, sWind, sWorkingday, sSeason, sWeather, sYear].forEach(el => {
        el.addEventListener('input', updateSimulation);
        el.addEventListener('change', updateSimulation);
    });

    // Run once on load
    updateSimulation();
}

/* -------------------------------------------------------------------------
   4. Image Modal Viewer
   ------------------------------------------------------------------------- */
function openModal(imgSrc) {
    const modal = document.getElementById('image-modal');
    const modalImg = document.getElementById('modal-img');
    if (modal && modalImg) {
        modalImg.src = imgSrc;
        modal.style.display = 'flex';
    }
}

function closeModal() {
    const modal = document.getElementById('image-modal');
    if (modal) {
        modal.style.display = 'none';
    }
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModal();
});

/* -------------------------------------------------------------------------
   5. Scroll Spy Navigation
   ------------------------------------------------------------------------- */
function initScrollSpy() {
    const sections = document.querySelectorAll('section');
    const navLinks = document.querySelectorAll('.nav-link');

    window.addEventListener('scroll', () => {
        let current = '';
        const scrollPosition = window.scrollY + 200;

        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.offsetHeight;
            if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    });
}

/* -------------------------------------------------------------------------
   6. Copy APA Report Text
   ------------------------------------------------------------------------- */
function copyApaText() {
    const textEl = document.getElementById('apa-report-text');
    if (!textEl) return;
    const text = textEl.innerText.trim();
    navigator.clipboard.writeText(text).then(() => {
        const btn = document.querySelector('.apa-copy-btn');
        if (btn) {
            const originalHtml = btn.innerHTML;
            btn.innerHTML = '<i class="fa-solid fa-check"></i> 已複製至剪貼簿！';
            btn.style.background = 'rgba(16, 185, 129, 0.3)';
            btn.style.color = '#34d399';
            setTimeout(() => {
                btn.innerHTML = originalHtml;
                btn.style.background = '';
                btn.style.color = '';
            }, 2500);
        }
    }).catch(err => {
        alert('複製失敗，請手動反白複製。');
    });
}

