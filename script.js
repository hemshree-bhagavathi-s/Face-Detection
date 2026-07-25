/**
 * Face Detection System using CNN
 * Frontend JavaScript Controller
 * 
 * Handles webcam stream capture, Base64 frame extraction, 300ms interval polling,
 * POST requests to Python Flask API (/predict), dynamic UI updates, and error handling.
 */

document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const startCamBtn = document.getElementById('startCamBtn');
    const stopCamBtn = document.getElementById('stopCamBtn');
    const webcamVideo = document.getElementById('webcamVideo');
    const frameCanvas = document.getElementById('frameCanvas');
    const webcamPlaceholder = document.getElementById('webcamPlaceholder');
    const viewportOverlay = document.getElementById('viewportOverlay');
    const targetBox = document.getElementById('targetBox');
    const apiLoadingOverlay = document.getElementById('apiLoadingOverlay');
    const liveIndicatorWrap = document.getElementById('liveIndicatorWrap');
    const statusDot = document.getElementById('statusDot');
    const liveText = document.getElementById('liveText');

    // Status & Stats Elements
    const detectionBadge = document.getElementById('detectionBadge');
    const badgeIcon = document.getElementById('badgeIcon');
    const badgeSpinner = document.getElementById('badgeSpinner');
    const badgeText = document.getElementById('badgeText');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceHint = document.getElementById('confidenceHint');
    const resVal = document.getElementById('resVal');
    const latencyVal = document.getElementById('latencyVal');
    const frameCountVal = document.getElementById('frameCountVal');
    const activeApiUrlDisplay = document.getElementById('activeApiUrlDisplay');
    const serverHealthDot = document.getElementById('serverHealthDot');
    const serverHealthText = document.getElementById('serverHealthText');

    // Banner & Modal Elements
    const errorBanner = document.getElementById('errorBanner');
    const errorMessage = document.getElementById('errorMessage');
    const closeErrorBtn = document.getElementById('closeErrorBtn');
    const settingsToggleBtn = document.getElementById('settingsToggleBtn');
    const settingsModal = document.getElementById('settingsModal');
    const closeModalBtn = document.getElementById('closeModalBtn');
    const saveSettingsBtn = document.getElementById('saveSettingsBtn');
    const apiUrlInput = document.getElementById('apiUrlInput');
    const frameRateSelect = document.getElementById('frameRateSelect');
    const testApiBtn = document.getElementById('testApiBtn');
    const testApiResult = document.getElementById('testApiResult');
    const demoModeToggle = document.getElementById('demoModeToggle');

    // State Variables
    let mediaStream = null;
    let captureInterval = null;
    let isProcessingFrame = false;
    let frameCount = 0;
    let apiUrl = localStorage.getItem('faceDetect_apiUrl') || 'http://127.0.0.1:5000/predict';
    let frameIntervalMs = parseInt(localStorage.getItem('faceDetect_interval') || '300', 10);
    let isDemoMode = false;
    let canvasCtx = frameCanvas.getContext('2d');

    // Initialize Settings UI State
    apiUrlInput.value = apiUrl;
    activeApiUrlDisplay.textContent = apiUrl;
    frameRateSelect.value = frameIntervalMs.toString();

    // Event Listeners
    startCamBtn.addEventListener('click', startWebcam);
    stopCamBtn.addEventListener('click', stopWebcam);
    closeErrorBtn.addEventListener('click', hideError);
    settingsToggleBtn.addEventListener('click', () => settingsModal.classList.remove('hidden'));
    closeModalBtn.addEventListener('click', () => settingsModal.classList.add('hidden'));
    saveSettingsBtn.addEventListener('click', saveSettings);
    testApiBtn.addEventListener('click', testApiConnection);
    demoModeToggle.addEventListener('change', (e) => {
        isDemoMode = e.target.checked;
        showToast(isDemoMode ? 'Demo Mode Activated (Simulated CNN Inference)' : 'Real API Mode Activated', 'info');
    });

    window.addEventListener('offline', () => {
        showError('No Internet Connection. Network requests will fail until reconnected.');
    });

    /**
     * Start Webcam Stream & Frame Sampling Loop
     */
    async function startWebcam() {
        hideError();

        // Check navigator compatibility
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            showError('Webcam access is not supported by your browser. Please use modern Chrome, Edge, or Firefox.');
            return;
        }

        try {
            // Update UI to requesting state
            startCamBtn.disabled = true;
            startCamBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> <span>Connecting...</span>`;

            // Request camera permissions with standard video constraints
            mediaStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                },
                audio: false
            });

            // Attach stream to video tag
            webcamVideo.srcObject = mediaStream;
            await webcamVideo.play();

            // Set canvas dimensions once metadata loaded
            const videoWidth = webcamVideo.videoWidth || 640;
            const videoHeight = webcamVideo.videoHeight || 480;
            frameCanvas.width = videoWidth;
            frameCanvas.height = videoHeight;
            resVal.textContent = `${videoWidth}x${videoHeight}`;

            // Update UI Layout
            webcamPlaceholder.classList.add('hidden');
            webcamVideo.classList.remove('hidden');
            viewportOverlay.classList.add('active');

            statusDot.className = 'pulse-dot green';
            liveText.textContent = 'LIVE';

            startCamBtn.innerHTML = `<i class="fa-solid fa-play"></i> <span>Start Camera</span>`;
            startCamBtn.disabled = true;
            stopCamBtn.disabled = false;

            // Reset Counters & Start Loop
            frameCount = 0;
            frameCountVal.textContent = '0';
            setStandbyStatus('Analyzing...');

            // Clear previous interval if any
            if (captureInterval) clearInterval(captureInterval);

            // Capture frame every 300 ms
            captureInterval = setInterval(captureAndSendFrame, frameIntervalMs);

        } catch (err) {
            startCamBtn.disabled = false;
            startCamBtn.innerHTML = `<i class="fa-solid fa-play"></i> <span>Start Camera</span>`;

            console.error('Camera Access Error:', err);
            handleCameraError(err);
        }
    }

    /**
     * Stop Webcam Stream & Reset Viewport
     */
    function stopWebcam() {
        if (captureInterval) {
            clearInterval(captureInterval);
            captureInterval = null;
        }

        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            mediaStream = null;
        }

        webcamVideo.pause();
        webcamVideo.srcObject = null;

        // Reset UI Components
        webcamVideo.classList.add('hidden');
        viewportOverlay.classList.remove('active');
        webcamPlaceholder.classList.remove('hidden');
        apiLoadingOverlay.classList.add('hidden');

        statusDot.className = 'pulse-dot grey';
        liveText.textContent = 'STANDBY';

        startCamBtn.disabled = false;
        stopCamBtn.disabled = true;

        resVal.textContent = '0x0';
        latencyVal.textContent = '0 ms';
        setStandbyStatus('Awaiting Input');
    }

    /**
     * Capture Frame from Canvas, convert to Base64, POST to Backend
     */
    async function captureAndSendFrame() {
        if (!mediaStream || isProcessingFrame) return;

        isProcessingFrame = true;
        apiLoadingOverlay.classList.remove('hidden');
        const startTime = performance.now();

        try {
            // Draw current video frame to canvas
            canvasCtx.drawImage(webcamVideo, 0, 0, frameCanvas.width, frameCanvas.height);

            // Export to JPEG Base64 data URL
            const base64Image = frameCanvas.toDataURL('image/jpeg', 0.85);

            let data;

            if (isDemoMode) {
                // Simulated CNN Inference Mode
                await new Promise(r => setTimeout(r, 90)); // Simulate 90ms latency
                const hasFace = Math.random() > 0.15; // 85% chance face detected in demo
                data = {
                    face_detected: hasFace,
                    confidence: hasFace ? (88.0 + Math.random() * 11.5) : (12.0 + Math.random() * 25.0)
                };
            } else {
                // Send Base64 frame to Flask backend
                const response = await fetch(apiUrl, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        image: base64Image
                    })
                });

                if (!response.ok) {
                    throw new Error(`Server returned HTTP ${response.status}: ${response.statusText}`);
                }

                data = await response.json();
            }

            const endTime = performance.now();
            const latency = Math.round(endTime - startTime);

            // Update stats
            frameCount++;
            frameCountVal.textContent = frameCount.toString();
            latencyVal.textContent = `${latency} ms`;
            updateServerHealth(true, 'Connected');

            // Render prediction results to UI
            renderDetectionResult(data);

        } catch (err) {
            console.warn('Frame submission failed:', err);
            updateServerHealth(false, 'Offline');
            
            // Render backend error feedback in badge
            detectionBadge.className = 'status-badge standby';
            badgeIcon.className = 'fa-solid fa-plug-circle-xmark';
            badgeSpinner.classList.add('hidden');
            badgeText.textContent = 'Backend Offline';
            
            confidenceValue.textContent = '0.0%';
            confidenceBar.style.width = '0%';
            confidenceHint.textContent = `Unable to reach API at ${apiUrl}. Ensure Flask is running or switch to Demo Mode.`;

        } finally {
            isProcessingFrame = false;
            apiLoadingOverlay.classList.add('hidden');
        }
    }

    /**
     * Update UI with Prediction Data from CNN Model
     */
    function renderDetectionResult(data) {
        if (!data || typeof data.face_detected === 'undefined') {
            showError('Invalid JSON response format received from Backend API.');
            return;
        }

        const isFaceDetected = Boolean(data.face_detected);
        
        // Parse confidence score (handles decimal 0.975 or percentage 97.5)
        let rawConfidence = parseFloat(data.confidence || 0);
        if (rawConfidence <= 1.0 && rawConfidence > 0) {
            rawConfidence *= 100;
        }
        const confidencePct = Math.min(100, Math.max(0, rawConfidence)).toFixed(1);

        badgeSpinner.classList.add('hidden');

        if (isFaceDetected) {
            // GREEN BADGE: Face Detected
            detectionBadge.className = 'status-badge detected';
            badgeIcon.className = 'fa-solid fa-circle-check';
            badgeText.textContent = 'Face Detected';

            targetBox.className = 'face-target-box detected';
            confidenceBar.style.background = 'linear-gradient(90deg, #3b82f6, #10b981)';
            confidenceBar.style.boxShadow = '0 0 12px rgba(16, 185, 129, 0.6)';
            confidenceHint.textContent = 'CNN Model recognizes facial features in live viewport.';
        } else {
            // RED BADGE: No Face Detected
            detectionBadge.className = 'status-badge not-detected';
            badgeIcon.className = 'fa-solid fa-circle-xmark';
            badgeText.textContent = 'No Face Detected';

            targetBox.className = 'face-target-box not-detected';
            confidenceBar.style.background = 'linear-gradient(90deg, #ef4444, #dc2626)';
            confidenceBar.style.boxShadow = '0 0 12px rgba(239, 68, 68, 0.6)';
            confidenceHint.textContent = 'No face detected in current frame. Align face with camera.';
        }

        // Update confidence bar width & text
        confidenceValue.textContent = `${confidencePct}%`;
        confidenceBar.style.width = `${confidencePct}%`;
    }

    /**
     * Set Status Badge to Standby / Waiting State
     */
    function setStandbyStatus(text) {
        detectionBadge.className = 'status-badge standby';
        badgeIcon.className = 'fa-solid fa-pause';
        badgeSpinner.classList.add('hidden');
        badgeText.textContent = text;
        targetBox.className = 'face-target-box';

        confidenceValue.textContent = '0.0%';
        confidenceBar.style.width = '0%';
        confidenceBar.style.background = 'linear-gradient(90deg, #3b82f6, #10b981)';
    }

    /**
     * Update Server Status Indicator
     */
    function updateServerHealth(isOnline, statusMsg) {
        if (isOnline) {
            serverHealthDot.className = 'dot-indicator green';
            serverHealthText.textContent = statusMsg || 'Connected';
            serverHealthText.style.color = '#34d399';
        } else {
            serverHealthDot.className = 'dot-indicator red';
            serverHealthText.textContent = statusMsg || 'Offline';
            serverHealthText.style.color = '#f87171';
        }
    }

    /**
     * Handle Camera Specific Access Errors
     */
    function handleCameraError(err) {
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
            showError('Camera Permission Denied! Please click the lock icon in your browser address bar and grant camera access.');
        } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
            showError('No webcam hardware detected. Please connect a webcam device.');
        } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
            showError('Camera is already in use by another application. Please close other video apps.');
        } else {
            showError(`Camera Error: ${err.message || 'Unable to open camera stream.'}`);
        }
    }

    /**
     * Show Error Toast Banner
     */
    function showError(msg) {
        errorMessage.textContent = msg;
        errorBanner.classList.remove('hidden');
    }

    function hideError() {
        errorBanner.classList.add('hidden');
    }

    /**
     * Toast Helper
     */
    function showToast(msg, type = 'info') {
        console.log(`[Toast ${type}]:`, msg);
    }

    /**
     * Save API Settings
     */
    function saveSettings() {
        const newUrl = apiUrlInput.value.trim();
        const newInterval = parseInt(frameRateSelect.value, 10);

        if (!newUrl) {
            alert('Please enter a valid API URL endpoint.');
            return;
        }

        apiUrl = newUrl;
        frameIntervalMs = newInterval;

        localStorage.setItem('faceDetect_apiUrl', apiUrl);
        localStorage.setItem('faceDetect_interval', frameIntervalMs);

        activeApiUrlDisplay.textContent = apiUrl;
        settingsModal.classList.add('hidden');

        // Restart loop if camera is active
        if (mediaStream) {
            if (captureInterval) clearInterval(captureInterval);
            captureInterval = setInterval(captureAndSendFrame, frameIntervalMs);
        }

        showToast('Settings saved successfully!');
    }

    /**
     * Test Backend API Endpoint Connection
     */
    async function testApiConnection() {
        const testUrl = apiUrlInput.value.trim();
        testApiResult.textContent = 'Testing connection...';
        testApiResult.style.color = '#94a3b8';

        try {
            // Test ping with dummy 1x1 base64 transparent gif frame
            const dummyFrame = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 4000);

            const res = await fetch(testUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: dummyFrame }),
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (res.ok) {
                testApiResult.textContent = `✓ Success! Server responded with HTTP ${res.status}`;
                testApiResult.style.color = '#34d399';
                updateServerHealth(true, 'Connected');
            } else {
                testApiResult.textContent = `⚠ Server reachable, but returned HTTP ${res.status}`;
                testApiResult.style.color = '#f59e0b';
            }
        } catch (err) {
            if (err.name === 'AbortError') {
                testApiResult.textContent = '✕ Connection timed out after 4 seconds.';
            } else {
                testApiResult.textContent = `✕ Failed to connect to ${testUrl}. Check Flask server.`;
            }
            testApiResult.style.color = '#f87171';
        }
    }
});
