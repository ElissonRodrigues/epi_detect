/**
 * EPI Monitor - Real-time PPE Detection Web Application
 */

class EPIMonitor {
    constructor() {
        // DOM Elements
        this.videoElement = document.getElementById('videoElement');
        this.outputCanvas = document.getElementById('outputCanvas');
        this.ctx = this.outputCanvas.getContext('2d');
        this.videoOverlay = document.getElementById('videoOverlay');
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.cameraSelect = document.getElementById('cameraSelect');
        this.connectionStatus = document.getElementById('connectionStatus');
        
        // Stats elements
        this.personCount = document.getElementById('personCount');
        this.compliantCount = document.getElementById('compliantCount');
        this.nonCompliantCount = document.getElementById('nonCompliantCount');
        this.fpsValue = document.getElementById('fpsValue');
        this.latencyValue = document.getElementById('latencyValue');
        this.personDetails = document.getElementById('personDetails');
        
        // State
        this.socket = null;
        this.stream = null;
        this.isRunning = false;
        this.frameCount = 0;
        this.lastFpsUpdate = Date.now();
        this.currentFps = 0;
        this.pendingFrame = false;
        
        // Bind methods
        this.init();
    }
    
    async init() {
        console.log('Initializing EPI Monitor...');
        
        // Setup event listeners
        this.startBtn.addEventListener('click', () => this.start());
        this.stopBtn.addEventListener('click', () => this.stop());
        this.videoOverlay.addEventListener('click', () => this.start());
        
        // Get available cameras
        await this.populateCameraList();
        
        // Setup WebSocket
        this.setupSocket();
    }
    
    setupSocket() {
        console.log('Connecting to WebSocket...');
        
        this.socket = io({
            transports: ['websocket'],
            reconnection: true,
            reconnectionAttempts: 5,
            reconnectionDelay: 1000
        });
        
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.updateConnectionStatus(true);
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.updateConnectionStatus(false);
        });
        
        this.socket.on('status', (data) => {
            console.log('Server status:', data.message);
        });
        
        this.socket.on('result', (data) => {
            this.handleResult(data);
        });
        
        this.socket.on('error', (data) => {
            console.error('Server error:', data.message);
        });
    }
    
    updateConnectionStatus(connected) {
        const dot = this.connectionStatus.querySelector('.status-dot');
        const text = this.connectionStatus.querySelector('.status-text');
        
        if (connected) {
            dot.classList.add('connected');
            text.textContent = 'Conectado';
        } else {
            dot.classList.remove('connected');
            text.textContent = 'Desconectado';
        }
    }
    
    async populateCameraList() {
        try {
            // Request permission first
            const tempStream = await navigator.mediaDevices.getUserMedia({ video: true });
            tempStream.getTracks().forEach(track => track.stop());
            
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(device => device.kind === 'videoinput');
            
            this.cameraSelect.innerHTML = '<option value="">Selecione a câmera...</option>';
            
            videoDevices.forEach((device, index) => {
                const option = document.createElement('option');
                option.value = device.deviceId;
                option.textContent = device.label || `Câmera ${index + 1}`;
                this.cameraSelect.appendChild(option);
            });
            
            // Auto-select first camera
            if (videoDevices.length > 0) {
                this.cameraSelect.value = videoDevices[0].deviceId;
            }
        } catch (error) {
            console.error('Error getting camera list:', error);
        }
    }
    
    async start() {
        if (this.isRunning) return;
        
        try {
            console.log('Starting camera...');
            
            const constraints = {
                video: {
                    deviceId: this.cameraSelect.value ? { exact: this.cameraSelect.value } : undefined,
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                }
            };
            
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.videoElement.srcObject = this.stream;
            
            await this.videoElement.play();
            
            // Set canvas size
            this.outputCanvas.width = this.videoElement.videoWidth || 1280;
            this.outputCanvas.height = this.videoElement.videoHeight || 720;
            
            // Update UI
            this.videoOverlay.classList.add('hidden');
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            
            this.isRunning = true;
            this.frameCount = 0;
            this.lastFpsUpdate = Date.now();
            
            // Start frame capture loop
            this.captureLoop();
            
            console.log('Camera started successfully');
        } catch (error) {
            console.error('Error starting camera:', error);
            alert('Erro ao acessar a câmera. Verifique as permissões.');
        }
    }
    
    stop() {
        console.log('Stopping camera...');
        
        this.isRunning = false;
        
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        this.videoElement.srcObject = null;
        this.videoOverlay.classList.remove('hidden');
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        
        // Clear stats
        this.personCount.textContent = '0';
        this.compliantCount.textContent = '0';
        this.nonCompliantCount.textContent = '0';
        this.fpsValue.textContent = '0';
        this.personDetails.innerHTML = '<p class="no-detections">Nenhuma pessoa detectada</p>';
    }
    
    captureLoop() {
        if (!this.isRunning) return;
        
        // Always draw local video first for immediate feedback
        if (this.videoElement.videoWidth > 0 && !this.pendingFrame) {
            // Capture frame for processing
            const captureCanvas = document.createElement('canvas');
            captureCanvas.width = this.videoElement.videoWidth || 640;
            captureCanvas.height = this.videoElement.videoHeight || 480;
            const captureCtx = captureCanvas.getContext('2d');
            
            captureCtx.drawImage(this.videoElement, 0, 0);
            
            // Convert to base64 and send
            const imageData = captureCanvas.toDataURL('image/jpeg', 0.7);
            
            if (this.socket && this.socket.connected) {
                this.pendingFrame = true;
                const startTime = Date.now();
                this.lastFrameTime = startTime;
                this.socket.emit('frame', { 
                    image: imageData,
                    timestamp: startTime
                });
            }
        }
        
        // Schedule next capture using requestAnimationFrame for smoother timing
        requestAnimationFrame(() => {
            setTimeout(() => this.captureLoop(), 50);
        });
    }
    
    handleResult(data) {
        this.pendingFrame = false;
        
        // Update FPS
        this.frameCount++;
        const now = Date.now();
        if (now - this.lastFpsUpdate >= 1000) {
            this.currentFps = this.frameCount;
            this.fpsValue.textContent = this.currentFps;
            this.frameCount = 0;
            this.lastFpsUpdate = now;
        }
        
        // Calculate latency
        if (data.timestamp) {
            const latency = now - data.timestamp;
            this.latencyValue.textContent = `${latency}ms`;
        }
        
        // Draw annotated frame immediately
        if (data.image) {
            const img = new Image();
            img.onload = () => {
                if (this.isRunning) {
                    this.outputCanvas.width = img.width;
                    this.outputCanvas.height = img.height;
                    this.ctx.drawImage(img, 0, 0);
                }
            };
            img.src = data.image;
        }
        
        // Update stats
        this.personCount.textContent = data.total_persons || 0;
        this.compliantCount.textContent = data.compliant || 0;
        this.nonCompliantCount.textContent = (data.total_persons || 0) - (data.compliant || 0);
        
        // Update person details
        this.updatePersonDetails(data.epi_status || []);
    }
    
    updatePersonDetails(epiStatus) {
        if (epiStatus.length === 0) {
            this.personDetails.innerHTML = '<p class="no-detections">Nenhuma pessoa detectada</p>';
            return;
        }
        
        let html = '';
        epiStatus.forEach((status, index) => {
            const hasHelmet = status.helmet;
            const hasVest = status.vest;
            const hasGloves = status.gloves;
            const epiCount = [hasHelmet, hasVest, hasGloves].filter(Boolean).length;
            
            let itemClass = 'person-item';
            let statusIcon = '';
            if (epiCount === 3) {
                itemClass += ' success';
                statusIcon = '✅';
            } else if (epiCount > 0) {
                itemClass += ' warning';
                statusIcon = '⚠️';
            } else {
                itemClass += ' danger';
                statusIcon = '❌';
            }
            
            html += `
                <div class="${itemClass}">
                    <div class="person-header">
                        <span class="person-status-icon">${statusIcon}</span>
                        <span class="person-name">Pessoa ${index + 1}</span>
                        <span class="epi-count">${epiCount}/3</span>
                    </div>
                    <div class="person-epi-list">
                        <div class="epi-item ${hasHelmet ? 'has-epi' : 'missing-epi'}">
                            <span class="epi-icon">🪖</span>
                            <span class="epi-label">Capacete</span>
                            <span class="epi-status">${hasHelmet ? '✓' : '✗'}</span>
                        </div>
                        <div class="epi-item ${hasVest ? 'has-epi' : 'missing-epi'}">
                            <span class="epi-icon">🦺</span>
                            <span class="epi-label">Colete</span>
                            <span class="epi-status">${hasVest ? '✓' : '✗'}</span>
                        </div>
                        <div class="epi-item ${hasGloves ? 'has-epi' : 'missing-epi'}">
                            <span class="epi-icon">🧤</span>
                            <span class="epi-label">Luvas</span>
                            <span class="epi-status">${hasGloves ? '✓' : '✗'}</span>
                        </div>
                    </div>
                </div>
            `;
        });
        
        this.personDetails.innerHTML = html;
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.epiMonitor = new EPIMonitor();
});
