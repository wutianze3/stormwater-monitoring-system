<template>
  <v-container fluid class="pa-6">
    <v-alert
      v-if="errorMessage"
      type="error"
      variant="tonal"
      closable
      class="mb-4"
      @click:close="errorMessage = ''"
    >
      {{ errorMessage }}
    </v-alert>

    <v-row>
      <v-col cols="12" md="8">
        <v-card color="surface" rounded="lg">
          <v-card-title class="pa-4 text-body-1 font-weight-bold">
            <v-icon class="mr-2" color="primary">mdi-camera</v-icon>
            Live Camera Feed
          </v-card-title>
          <v-divider />

          <div ref="stage" class="camera-stage">
            <video
              v-show="cameraActive"
              ref="video"
              autoplay
              muted
              playsinline
              class="camera-video"
              @loadedmetadata="handleVideoReady"
            />
            <canvas
              v-show="cameraActive || uploadedPreview"
              ref="overlay"
              class="detection-overlay"
            />
            <img
              v-if="!cameraActive && uploadedPreview"
              ref="uploadedImage"
              :src="uploadedPreview"
              class="camera-video"
              alt="Uploaded stormwater sample"
              @load="drawDetections"
            />

            <div v-if="!cameraActive && !uploadedPreview" class="camera-placeholder">
              <v-icon size="80" color="grey-darken-1">mdi-camera-off</v-icon>
              <div class="text-medium-emphasis mt-4">Camera is not running</div>
              <div class="text-caption text-medium-emphasis mt-1">
                Start the camera or upload an image for waste detection
              </div>
            </div>

            <div v-if="analysing" class="analysis-indicator">
              <v-progress-circular indeterminate size="18" width="2" class="mr-2" />
              Analysing frame
            </div>
          </div>

          <canvas ref="captureCanvas" class="d-none" />
          <input
            ref="fileInput"
            type="file"
            accept="image/jpeg,image/png,image/webp"
            class="d-none"
            @change="handleFileUpload"
          />

          <v-divider />
          <v-card-actions class="pa-4 flex-wrap ga-2">
            <v-btn
              :color="cameraActive ? 'error' : 'primary'"
              :prepend-icon="cameraActive ? 'mdi-camera-off' : 'mdi-camera'"
              variant="flat"
              @click="cameraActive ? stopCamera() : startCamera()"
            >
              {{ cameraActive ? 'Stop camera' : 'Start camera' }}
            </v-btn>
            <v-btn
              prepend-icon="mdi-eye-check"
              variant="tonal"
              :disabled="!cameraActive || analysing"
              @click="analyseCurrentFrame"
            >
              Analyse now
            </v-btn>
            <v-btn
              prepend-icon="mdi-upload"
              variant="tonal"
              :disabled="analysing"
              @click="$refs.fileInput.click()"
            >
              Upload image
            </v-btn>
            <v-spacer />
            <v-switch
              v-model="autoAnalyse"
              label="Auto analyse"
              color="primary"
              density="compact"
              hide-details
              :disabled="!cameraActive"
              @update:model-value="configureAutoAnalysis"
            />
          </v-card-actions>
        </v-card>
      </v-col>

      <v-col cols="12" md="4">
        <v-card color="surface" rounded="lg" height="100%">
          <v-card-title class="pa-4 text-body-1 font-weight-bold">
            <v-icon class="mr-2" color="primary">mdi-eye-check</v-icon>
            Detection Results
          </v-card-title>
          <v-divider />
          <v-card-text>
            <div class="mb-4">
              <div class="text-medium-emphasis text-caption mb-1">Visible Litter Index (not water quality)</div>
              <div class="d-flex align-center justify-space-between">
                <span class="text-h4 font-weight-bold">{{ result?.score ?? '—' }}</span>
                <v-chip size="small" :color="riskColor" variant="tonal">
                  {{ riskLabel }}
                </v-chip>
              </div>
            </div>

            <v-divider class="mb-4" />

            <div class="mb-4">
              <div class="text-medium-emphasis text-caption mb-1">Turbidity (Visual)</div>
              <div class="d-flex align-center justify-space-between">
                <span class="text-medium-emphasis text-body-2">Brown/low-contrast cue</span>
                <v-chip size="small" :color="factorColor(result?.factors?.turbidity)" variant="tonal">
                  {{ factorPercent(result?.factors?.turbidity) }}
                </v-chip>
              </div>
            </div>

            <v-divider class="mb-4" />

            <div class="mb-4">
              <div class="text-medium-emphasis text-caption mb-1">Visual Anomalies</div>
              <div class="d-flex align-center justify-space-between">
                <span class="text-medium-emphasis text-body-2">Detected objects</span>
                <v-chip size="small" :color="result?.detectionCount ? 'warning' : 'success'" variant="tonal">
                  {{ result ? result.detectionCount : 'Awaiting feed' }}
                </v-chip>
              </div>
            </div>

            <v-divider class="mb-4" />

            <div class="text-caption text-medium-emphasis mb-1">Analysis details</div>
            <v-list bg-color="transparent" density="compact">
              <v-list-item
                prepend-icon="mdi-gauge"
                title="Highest object confidence (not accuracy)"
                :subtitle="result ? `${result.confidence}%` : 'Awaiting analysis'"
                density="compact"
              />
              <v-list-item
                prepend-icon="mdi-image-filter-center-focus"
                title="Detection method"
                :subtitle="result?.source || 'YOLOv8 waste detection'"
                density="compact"
              />
            </v-list>

            <v-alert type="info" variant="tonal" class="mt-4" density="compact">
              Visible waste screening only. Organic objects and unknown objects do not increase the litter index. Results do not establish water safety.
            </v-alert>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
const API_BASE = import.meta.env.VITE_API_BASE_URL
  || `${window.location.protocol}//${window.location.hostname}:8000`

export default {
  name: 'Camera',

  data() {
    return {
      stream: null,
      cameraActive: false,
      analysing: false,
      autoAnalyse: true,
      analysisTimer: null,
      result: null,
      errorMessage: '',
      uploadedPreview: '',
    }
  },

  computed: {
    riskColor() {
      if (!this.result) return 'grey'
      if (this.result.score >= 68) return 'error'
      if (this.result.score >= 38) return 'warning'
      return 'success'
    },

    riskLabel() {
      if (!this.result) return 'Awaiting feed'
      if (this.result.score >= 68) return 'High'
      if (this.result.score >= 38) return 'Moderate'
      return 'Low'
    },
  },

  mounted() {
    window.addEventListener('resize', this.drawDetections)
  },

  beforeUnmount() {
    window.removeEventListener('resize', this.drawDetections)
    this.stopCamera()
    this.releaseUploadedPreview()
  },

  methods: {
    async startCamera() {
      this.errorMessage = ''
      this.result = null
      this.releaseUploadedPreview()
      if (!navigator.mediaDevices?.getUserMedia) {
        this.errorMessage = 'Camera access is not supported by this browser.'
        return
      }

      try {
        this.stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: 'environment' } },
          audio: false,
        })
        this.cameraActive = true
        await this.$nextTick()
        this.$refs.video.srcObject = this.stream
        await this.$refs.video.play()
        this.configureAutoAnalysis()
      } catch (error) {
        this.errorMessage = 'Unable to open the camera. Check the browser camera permission.'
        console.error('camera access failed', error)
        this.stopCamera()
      }
    },

    stopCamera() {
      if (this.analysisTimer) clearInterval(this.analysisTimer)
      this.analysisTimer = null
      this.stream?.getTracks().forEach((track) => track.stop())
      this.stream = null
      this.cameraActive = false
      if (this.$refs.video) this.$refs.video.srcObject = null
      this.clearOverlay()
    },

    handleVideoReady() {
      this.drawDetections()
      if (this.autoAnalyse) this.analyseCurrentFrame()
    },

    configureAutoAnalysis() {
      if (this.analysisTimer) clearInterval(this.analysisTimer)
      this.analysisTimer = null
      if (this.cameraActive && this.autoAnalyse) {
        this.analysisTimer = setInterval(() => {
          if (!this.analysing) this.analyseCurrentFrame()
        }, 2500)
      }
    },

    async analyseCurrentFrame() {
      const video = this.$refs.video
      if (!this.cameraActive || !video || video.readyState < 2 || this.analysing) return

      const canvas = this.$refs.captureCanvas
      const maxWidth = 1280
      const scale = Math.min(1, maxWidth / video.videoWidth)
      canvas.width = Math.round(video.videoWidth * scale)
      canvas.height = Math.round(video.videoHeight * scale)
      canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height)
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.86))
      if (blob) await this.sendForAnalysis(blob, 'camera-frame.jpg')
    },

    async handleFileUpload(event) {
      const file = event.target.files?.[0]
      event.target.value = ''
      if (!file) return
      if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
        this.errorMessage = 'Choose a JPG, PNG, or WEBP image.'
        return
      }
      this.stopCamera()
      this.result = null
      this.releaseUploadedPreview()
      this.uploadedPreview = URL.createObjectURL(file)
      await this.sendForAnalysis(file, file.name)
    },

    async sendForAnalysis(blob, filename) {
      this.analysing = true
      this.errorMessage = ''
      try {
        const body = new FormData()
        body.append('image', blob, filename)
        body.append('context', 'stormwater-camera')
        const response = await fetch(`${API_BASE}/api/vision/analyze`, {
          method: 'POST',
          body,
          signal: AbortSignal.timeout(30000),
        })
        if (!response.ok) {
          const failure = await response.json().catch(() => ({}))
          throw new Error(failure.detail || `Analysis failed (${response.status})`)
        }
        this.result = await response.json()
        await this.$nextTick()
        this.drawDetections()
      } catch (error) {
        this.result = null
        this.clearOverlay()
        this.errorMessage = `Waste analysis failed: ${error.message}`
        console.error('vision analysis failed', error)
      } finally {
        this.analysing = false
      }
    },

    drawDetections() {
      const canvas = this.$refs.overlay
      const stage = this.$refs.stage
      if (!canvas || !stage) return
      const width = stage.clientWidth
      const height = stage.clientHeight
      const dpr = window.devicePixelRatio || 1
      canvas.width = Math.round(width * dpr)
      canvas.height = Math.round(height * dpr)
      canvas.style.width = `${width}px`
      canvas.style.height = `${height}px`
      const context = canvas.getContext('2d')
      context.scale(dpr, dpr)
      context.clearRect(0, 0, width, height)

      if (!this.result?.detections?.length) return
      const source = this.cameraActive ? this.$refs.video : this.$refs.uploadedImage
      const sourceWidth = source?.videoWidth || source?.naturalWidth
      const sourceHeight = source?.videoHeight || source?.naturalHeight
      if (!sourceWidth || !sourceHeight) return
      const imageScale = Math.min(width / sourceWidth, height / sourceHeight)
      const shownWidth = sourceWidth * imageScale
      const shownHeight = sourceHeight * imageScale
      const offsetX = (width - shownWidth) / 2
      const offsetY = (height - shownHeight) / 2

      this.result.detections.forEach((box) => {
        const x = offsetX + box.x1 * shownWidth
        const y = offsetY + box.y1 * shownHeight
        const boxWidth = (box.x2 - box.x1) * shownWidth
        const boxHeight = (box.y2 - box.y1) * shownHeight
        const label = `${box.label} ${Math.round(box.confidence * 100)}%`
        context.strokeStyle = '#00BCD4'
        context.lineWidth = 3
        context.strokeRect(x, y, boxWidth, boxHeight)
        context.font = '600 13px sans-serif'
        const labelWidth = context.measureText(label).width + 12
        const labelY = Math.max(0, y - 24)
        context.fillStyle = '#00BCD4'
        context.fillRect(x, labelY, labelWidth, 23)
        context.fillStyle = '#06131f'
        context.fillText(label, x + 6, labelY + 16)
      })
    },

    clearOverlay() {
      const canvas = this.$refs.overlay
      if (canvas) canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
    },

    releaseUploadedPreview() {
      if (this.uploadedPreview) URL.revokeObjectURL(this.uploadedPreview)
      this.uploadedPreview = ''
    },

    factorPercent(value) {
      return Number.isFinite(value) ? `${Math.round(value * 100)}%` : 'Not measured'
    },

    factorColor(value) {
      if (!Number.isFinite(value)) return 'grey'
      if (value >= 0.68) return 'error'
      if (value >= 0.38) return 'warning'
      return 'success'
    },
  },
}
</script>

<style scoped>
.camera-stage {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: min(58vh, 520px);
  min-height: 360px;
  overflow: hidden;
  background: #07101d;
}

.camera-video {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.camera-placeholder {
  text-align: center;
}

.detection-overlay {
  z-index: 1;
  position: absolute;
  inset: 0;
  pointer-events: none;
}

.analysis-indicator {
  position: absolute;
  right: 16px;
  bottom: 16px;
  display: flex;
  align-items: center;
  padding: 8px 12px;
  border-radius: 999px;
  color: white;
  background: rgba(3, 20, 31, 0.82);
  backdrop-filter: blur(8px);
}

@media (max-width: 600px) {
  .camera-stage {
    height: 420px;
    min-height: 300px;
  }
}
</style>
