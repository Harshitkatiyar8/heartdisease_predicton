document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const navToggle = document.getElementById('nav-toggle');
    const navMenu = document.getElementById('nav-menu');
    const navLinks = document.querySelectorAll('.nav-link');
    const form = document.getElementById('prediction-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = document.getElementById('btn-text');
    const btnSpinner = document.getElementById('btn-spinner');
    const errorAlert = document.getElementById('error-alert');
    const errorAlertText = document.getElementById('error-alert-text');
    const resultContainer = document.getElementById('result-container');

    /* ==================================================
       NAVIGATION & MOBILE MENU
       ================================================== */
    if (navToggle && navMenu) {
        navToggle.addEventListener('click', () => {
            navMenu.classList.toggle('open');
        });
    }

    // Close mobile menu when a nav link is clicked
    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            if (navMenu && navMenu.classList.contains('open')) {
                navMenu.classList.remove('open');
            }
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        });
    });

    /* ==================================================
       FORM SUBMISSION & VALIDATION
       ================================================== */
    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            // Clear previous errors
            hideError();

            // Extract form field values
            const formData = new FormData(form);
            const payload = {};
            formData.forEach((value, key) => {
                payload[key] = value;
            });

            // Client-side Numerical Range Validation
            const age = parseInt(payload.age, 10);
            const restingBp = parseInt(payload.resting_bp, 10);
            const cholesterol = parseInt(payload.cholesterol, 10);
            const maxHr = parseInt(payload.max_hr, 10);
            const oldpeak = parseFloat(payload.oldpeak);

            if (isNaN(age) || age < 18 || age > 120) {
                showError('Please enter a valid age between 18 and 120 years.');
                return;
            }
            if (isNaN(restingBp) || restingBp < 60 || restingBp > 240) {
                showError('Please enter a valid resting blood pressure between 60 and 240 mm Hg.');
                return;
            }
            if (isNaN(cholesterol) || cholesterol < 0 || cholesterol > 600) {
                showError('Please enter a valid serum cholesterol level between 0 and 600 mg/dl.');
                return;
            }
            if (isNaN(maxHr) || maxHr < 40 || maxHr > 220) {
                showError('Please enter a valid maximum heart rate between 40 and 220 bpm.');
                return;
            }
            if (isNaN(oldpeak) || oldpeak < -2.6 || oldpeak > 10.0) {
                showError('Please enter a valid Oldpeak ST depression value between -2.6 and 10.0.');
                return;
            }

            // Set Loading state (prevents duplicate submissions)
            setLoading(true);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    },
                    body: JSON.stringify(payload)
                });

                const data = await response.json();

                if (!response.ok || !data.success) {
                    const msg = data.error || (data.errors ? data.errors.join(' ') : 'Please enter valid information in all required fields.');
                    showError(msg);
                    return;
                }

                // Render prediction result card
                renderResult(data.prediction, data.result_text, data.risk_probability);

            } catch (err) {
                showError('Unable to connect to the prediction server. Please check your network connection.');
                console.error('Fetch error:', err);
            } finally {
                setLoading(false);
            }
        });
    }

    /* ==================================================
       RESET BUTTON HANDLER ("Check Another Patient")
       ================================================== */
    document.addEventListener('click', (e) => {
        if (e.target && (e.target.id === 'reset-btn' || e.target.closest('#reset-btn'))) {
            if (form) {
                form.reset();
            }
            if (resultContainer) {
                resultContainer.classList.add('hidden');
                resultContainer.innerHTML = '';
            }
            hideError();
            
            // Scroll back to prediction form top
            const predSection = document.getElementById('prediction');
            if (predSection) {
                predSection.scrollIntoView({ behavior: 'smooth' });
            }
            
            // Focus on first input
            const firstInput = document.getElementById('age');
            if (firstInput) {
                firstInput.focus();
            }
        }
    });

    /* ==================================================
       HELPER FUNCTIONS
       ================================================== */
    function showError(message) {
        if (errorAlertText && errorAlert) {
            errorAlertText.textContent = message;
            errorAlert.classList.remove('hidden');
            errorAlert.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }

    function hideError() {
        if (errorAlert) {
            errorAlert.classList.add('hidden');
        }
    }

    function setLoading(isLoading) {
        if (!submitBtn || !btnText || !btnSpinner) return;

        if (isLoading) {
            submitBtn.disabled = true;
            btnText.textContent = 'Analyzing health parameters...';
            btnSpinner.classList.remove('hidden');
        } else {
            submitBtn.disabled = false;
            btnText.textContent = 'Predict Heart Disease Risk';
            btnSpinner.classList.add('hidden');
        }
    }

    function renderResult(prediction, resultText, riskProbability) {
        if (!resultContainer) return;

        const isRiskDetected = prediction === 1;
        const cardClass = isRiskDetected ? 'state-risk' : 'state-norisk';
        const fillClass = isRiskDetected ? 'fill-risk' : 'fill-norisk';
        
        const badgeSvg = isRiskDetected ? `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="8" x2="12" y2="12"></line>
                <line x1="12" y1="16" x2="12.01" y2="16"></line>
            </svg>
        ` : `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                <polyline points="22 4 12 14.01 9 11.01"></polyline>
            </svg>
        `;

        const descText = isRiskDetected
            ? 'The Logistic Regression model detected patterns associated with elevated heart disease risk based on the provided clinical features.'
            : 'The Logistic Regression model detected no significant indicators of heart disease risk based on the provided clinical features.';

        const probBarHtml = (riskProbability !== null && riskProbability !== undefined) ? `
            <div class="result-probability-box">
                <div class="prob-header">
                    <span>Prediction Probability</span>
                    <span class="prob-value">${riskProbability}%</span>
                </div>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill ${fillClass}" style="width: ${riskProbability}%;"></div>
                </div>
            </div>
        ` : '';

        resultContainer.innerHTML = `
            <div class="result-card ${cardClass}">
                <div class="result-top">
                    <div class="result-badge-icon">
                        ${badgeSvg}
                    </div>
                    <div class="result-heading-box">
                        <span class="result-label">Prediction Result</span>
                        <h3 class="result-status-title">${resultText}</h3>
                    </div>
                </div>

                ${probBarHtml}

                <p class="result-explanation">${descText}</p>

                <div class="result-actions">
                    <button type="button" id="reset-btn" class="btn btn-secondary">
                        <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="1 4 1 10 7 10"></polyline>
                            <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"></path>
                        </svg>
                        <span>Check Another Patient</span>
                    </button>
                </div>
            </div>
        `;

        resultContainer.classList.remove('hidden');
        resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
});
