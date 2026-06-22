<template>
  <div class="app-shell py-6">
    <div class="page-container space-y-6">
      <header class="glass-panel rounded-[2rem] px-6 py-5 lg:px-8">
        <div class="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <button class="ghost-btn" @click="$router.push('/')">Назад к панели</button>
            <p class="mt-4 text-xs uppercase tracking-[0.28em] text-slate-400">SpatialScope · Slide workspace</p>
            <h1 class="mt-3 text-3xl font-semibold text-white">{{ slideName || 'Просмотр слайда' }}</h1>
            <p class="mt-2 text-sm text-slate-400">
              Настройка ROI для инференса, запуск обучения и просмотр данных выбранного Xenium-слайда.
            </p>
          </div>
          <div class="flex flex-wrap items-center gap-3">
            <span class="status-pill" :class="statusClass(statusTextRaw)">{{ statusText }}</span>
            <span class="badge">Slide ID: {{ slideId }}</span>
          </div>
        </div>
      </header>

      <div v-if="errorMessage" class="rounded-3xl border border-rose-400/20 bg-rose-400/10 px-5 py-4 text-sm text-rose-200">
        {{ errorMessage }}
      </div>

      <section class="grid grid-cols-1 gap-6 xl:grid-cols-[1.65fr_1fr]">
        <div class="glass-panel rounded-[2rem] p-4 lg:p-6">
          <div class="mb-4 flex items-center justify-between">
            <div>
              <h2 class="section-title">Область просмотра WSI</h2>
              <p class="section-subtitle mt-2 text-sm">Интерактивный просмотр whole-slide image через OpenSeadragon.</p>
            </div>
            <div class="flex flex-wrap gap-2 text-xs text-slate-400">
              <span class="badge">{{ slideMeta.width || '—' }} × {{ slideMeta.height || '—' }}</span>
              <span class="badge">MPP {{ slideMeta.mpp || '—' }}</span>
            </div>
          </div>
          <div class="viewer-frame glass-panel-soft">
            <div id="wsi-viewer" style="width: 100%; height: 60vh;"></div>
          </div>
        </div>

        <div class="space-y-6">
          <article class="glass-panel rounded-[2rem] p-6">
            <div class="flex items-center justify-between gap-3">
              <div>
                <h2 class="section-title">Инференс по ROI</h2>
                <p class="section-subtitle mt-2 text-sm">Доступно для administrator, researcher и clinician.</p>
              </div>
              <span class="badge">{{ canRunInference ? 'Доступно' : 'Недоступно' }}</span>
            </div>

            <div class="mt-5 grid gap-3 sm:grid-cols-2">
              <div>
                <label class="mb-2 block text-sm font-medium text-slate-300">X</label>
                <input v-model.number="inferenceRegion.x" class="input-control" type="number">
              </div>
              <div>
                <label class="mb-2 block text-sm font-medium text-slate-300">Y</label>
                <input v-model.number="inferenceRegion.y" class="input-control" type="number">
              </div>
              <div>
                <label class="mb-2 block text-sm font-medium text-slate-300">Ширина ROI</label>
                <input v-model.number="inferenceRegion.width" class="input-control" type="number">
              </div>
              <div>
                <label class="mb-2 block text-sm font-medium text-slate-300">Высота ROI</label>
                <input v-model.number="inferenceRegion.height" class="input-control" type="number">
              </div>
              <div>
                <label class="mb-2 block text-sm font-medium text-slate-300">Patch size</label>
                <input v-model.number="patchSize" class="input-control" type="number">
              </div>
              <div>
                <label class="mb-2 block text-sm font-medium text-slate-300">Target gene</label>
                <input v-model="targetGene" class="input-control" type="text" placeholder="CD3D">
              </div>
              <div class="sm:col-span-2">
                <label class="mb-2 block text-sm font-medium text-slate-300">Пресет модели</label>
                <select v-model="selectedPreset" class="select-control">
                  <option v-for="preset in presets" :key="preset.id" :value="preset.id">
                    {{ preset.name }}
                  </option>
                </select>
              </div>
            </div>

            <button class="primary-btn mt-5 w-full" :disabled="!canRunInference || inferenceLoading" @click="runInference">
              {{ inferenceLoading ? 'Выполняется инференс...' : 'Запустить инференс' }}
            </button>
          </article>

          <article class="glass-panel rounded-[2rem] p-6">
            <div class="flex items-center justify-between gap-3">
              <div>
                <h2 class="section-title">Обучение модели</h2>
                <p class="section-subtitle mt-2 text-sm">Доступно для administrator и researcher.</p>
              </div>
              <span class="badge">{{ canTrain ? 'Доступно' : 'Недоступно' }}</span>
            </div>

            <div class="mt-5 grid gap-3 sm:grid-cols-2">
              <div>
                <label class="mb-2 block text-sm font-medium text-slate-300">X</label>
                <input v-model.number="trainingRegion.x" class="input-control" type="number">
              </div>
              <div>
                <label class="mb-2 block text-sm font-medium text-slate-300">Y</label>
                <input v-model.number="trainingRegion.y" class="input-control" type="number">
              </div>
              <div>
                <label class="mb-2 block text-sm font-medium text-slate-300">Ширина ROI</label>
                <input v-model.number="trainingRegion.width" class="input-control" type="number">
              </div>
              <div>
                <label class="mb-2 block text-sm font-medium text-slate-300">Высота ROI</label>
                <input v-model.number="trainingRegion.height" class="input-control" type="number">
              </div>
              <div>
                <label class="mb-2 block text-sm font-medium text-slate-300">Эпохи</label>
                <input v-model.number="epochs" class="input-control" type="number">
              </div>
              <div>
                <label class="mb-2 block text-sm font-medium text-slate-300">Макс. патчей</label>
                <input v-model.number="maxPatches" class="input-control" type="number">
              </div>
            </div>

            <button class="secondary-btn mt-5 w-full" :disabled="!canTrain || trainingLoading" @click="startTraining">
              {{ trainingLoading ? 'Запуск обучения...' : 'Запустить обучение' }}
            </button>

            <div v-if="jobId" class="mt-4 rounded-2xl border border-white/8 bg-white/5 p-4 text-sm text-slate-300">
              <p><span class="text-slate-500">Job ID:</span> {{ jobId }}</p>
              <p class="mt-1"><span class="text-slate-500">Статус:</span> {{ jobStatus || 'running' }}</p>
            </div>
          </article>

          <article class="glass-panel rounded-[2rem] p-6">
            <div class="flex items-center justify-between gap-3">
              <div>
                <h2 class="section-title">История отчётов</h2>
                <p class="section-subtitle mt-2 text-sm">Результаты инференса для текущего слайда.</p>
              </div>
              <button class="ghost-btn" @click="loadReports">Обновить</button>
            </div>

            <div v-if="reports.length === 0" class="mt-4 text-sm text-slate-400">Отчёты для этого слайда пока отсутствуют.</div>
            <div v-else class="mt-4 space-y-3">
              <div v-for="report in reports" :key="report.id" class="rounded-2xl border border-white/8 bg-white/5 p-4">
                <div class="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                  <div>
                    <p class="font-medium text-white">Отчёт {{ report.id }}</p>
                    <p class="mt-1 text-xs text-slate-500">{{ formatDate(report.created_at) }}</p>
                  </div>
                  <button class="ghost-btn" @click="openReport(report.id)">Открыть отчёт</button>
                </div>
              </div>
            </div>
          </article>
        </div>
      </section>
    </div>
  </div>
</template>

<script>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import OpenSeadragon from 'openseadragon'
import api, { buildApiUrl } from '../api'

export default {
  name: 'SlideViewerView',
  setup() {
    const route = useRoute()
    const router = useRouter()
    const slideId = route.params.id
    const user = ref(JSON.parse(localStorage.getItem('user') || '{"role":"clinician"}'))
    const slideName = ref('')
    const slideMeta = ref({})
    const reports = ref([])
    const presets = ref([])
    const selectedPreset = ref('')
    const patchSize = ref(512)
    const maxPatches = ref(20)
    const epochs = ref(30)
    const targetGene = ref('CD3D')
    const inferenceRegion = ref({ x: 0, y: 0, width: 1024, height: 1024 })
    const trainingRegion = ref({ x: 0, y: 0, width: 2048, height: 2048 })
    const statusText = ref('Загрузка')
    const statusTextRaw = ref('queued')
    const errorMessage = ref('')
    const jobId = ref('')
    const jobStatus = ref('')
    const inferenceLoading = ref(false)
    const trainingLoading = ref(false)
    const viewer = ref(null)

    let pollTimer = null

    const role = computed(() => user.value?.role || 'clinician')
    const canRunInference = computed(() => ['admin', 'researcher', 'clinician'].includes(role.value))
    const canTrain = computed(() => ['admin', 'researcher'].includes(role.value))

    const statusClass = (status) => `status-${status || 'queued'}`

    const formatDate = (value) => new Date(value).toLocaleString('ru-RU')

    const loadReports = async () => {
      const response = await api.get(`/reports/slide/${slideId}`)
      reports.value = response.data.data || []
    }

    const loadPresets = async () => {
      const response = await api.get('/presets')
      presets.value = response.data.data || []
      selectedPreset.value = presets.value.find((preset) => preset.is_active)?.id || presets.value[0]?.id || ''
    }

    const loadSlide = async () => {
      const response = await api.get(`/slides/${slideId}`)
      const slide = response.data.data
      slideName.value = slide.patient || 'Слайд'
      slideMeta.value = {
        width: slide.width,
        height: slide.height,
        mpp: slide.mpp,
      }

      if (slide.width && slide.height) {
        inferenceRegion.value = {
          x: 0,
          y: 0,
          width: Math.min(1024, slide.width),
          height: Math.min(1024, slide.height),
        }
        trainingRegion.value = {
          x: 0,
          y: 0,
          width: Math.min(2048, slide.width),
          height: Math.min(2048, slide.height),
        }
      }

      statusTextRaw.value = slide.status || 'ready'
      statusText.value = slide.status || 'ready'
    }

    const initViewer = () => {
      if (viewer.value) {
        viewer.value.destroy()
      }

      const width = Number(slideMeta.value.width || 1)
      const height = Number(slideMeta.value.height || 1)
      const patchSize = 1024

      viewer.value = OpenSeadragon({
        id: 'wsi-viewer',
        tileSources: {
          width,
          height,
          tileWidth: patchSize,
          tileHeight: patchSize,
          minLevel: 0,
          maxLevel: Math.max(Math.ceil(Math.log2(Math.max(width, height))), 0),
          getTileUrl(level, x, y) {
            const scale = 2 ** Math.max(this.maxLevel - level, 0)
            const regionX = Math.floor(x * patchSize * scale)
            const regionY = Math.floor(y * patchSize * scale)
            const regionWidth = Math.min(Math.floor(patchSize * scale), Math.max(width - regionX, 1))
            const regionHeight = Math.min(Math.floor(patchSize * scale), Math.max(height - regionY, 1))
            const token = localStorage.getItem('token')
            const url = new URL(buildApiUrl(`/slides/${slideId}/patch`), window.location.origin)
            url.searchParams.set('x', String(regionX))
            url.searchParams.set('y', String(regionY))
            url.searchParams.set('width', String(regionWidth))
            url.searchParams.set('height', String(regionHeight))
            if (token) {
              url.searchParams.set('token', token)
            }
            return url.toString()
          },
          getTileAjaxHeaders() {
            const token = localStorage.getItem('token')
            return token ? { Authorization: `Bearer ${token}` } : {}
          },
        },
        showNavigator: true,
        defaultZoomLevel: 0.75,
        minZoomLevel: 0.1,
        maxZoomLevel: 12,
      })
    }

    const pollJob = (id) => {
      if (pollTimer) clearInterval(pollTimer)
      pollTimer = setInterval(async () => {
        try {
          const response = await api.get(`/training/status/${id}`)
          jobStatus.value = response.data.data.status
          statusTextRaw.value = response.data.data.status
          statusText.value = response.data.data.status

          if (['success', 'failed'].includes(jobStatus.value) || jobStatus.value?.startsWith('failed')) {
            clearInterval(pollTimer)
            trainingLoading.value = false
            await loadSlide()
          }
        } catch (error) {
          clearInterval(pollTimer)
          trainingLoading.value = false
        }
      }, 3000)
    }

    const runInference = async () => {
      if (!canRunInference.value) return
      inferenceLoading.value = true
      errorMessage.value = ''
      statusText.value = 'inference running'
      statusTextRaw.value = 'running'

      try {
        const response = await api.post('/inference/run', {
          slide_id: slideId,
          x: inferenceRegion.value.x,
          y: inferenceRegion.value.y,
          patch_size: patchSize.value,
          preset_id: selectedPreset.value,
          inference_region: inferenceRegion.value,
          target_gene: targetGene.value,
          include_interpretation: true,
          regional_spots: [],
        })

        const reportId = response.data.data.report_id
        await loadReports()
        router.push(`/results/${reportId}`)
      } catch (error) {
        errorMessage.value = error.response?.data?.detail || 'Не удалось выполнить инференс'
        statusText.value = 'error'
        statusTextRaw.value = 'failed'
      } finally {
        inferenceLoading.value = false
      }
    }

    const startTraining = async () => {
      if (!canTrain.value) return
      trainingLoading.value = true
      errorMessage.value = ''
      statusText.value = 'training running'
      statusTextRaw.value = 'processing'

      try {
        const response = await api.post('/training/start', {
          slide_id: slideId,
          preset_id: selectedPreset.value,
          epochs: epochs.value,
          patch_size: patchSize.value,
          max_patches: maxPatches.value,
          training_region: trainingRegion.value,
        })

        jobId.value = response.data.job_id
        jobStatus.value = 'running'
        pollJob(jobId.value)
      } catch (error) {
        trainingLoading.value = false
        statusText.value = 'error'
        statusTextRaw.value = 'failed'
        errorMessage.value = error.response?.data?.detail || 'Не удалось запустить обучение'
      }
    }

    const openReport = (reportId) => {
      router.push(`/results/${reportId}`)
    }

    onMounted(async () => {
      try {
        await Promise.all([loadSlide(), loadPresets(), loadReports()])
        initViewer()
      } catch (error) {
        errorMessage.value = error.response?.data?.detail || 'Не удалось загрузить рабочее пространство слайда'
      }
    })

    onBeforeUnmount(() => {
      if (pollTimer) clearInterval(pollTimer)
      if (viewer.value) viewer.value.destroy()
    })

    return {
      slideId,
      slideName,
      slideMeta,
      reports,
      presets,
      selectedPreset,
      patchSize,
      maxPatches,
      epochs,
      targetGene,
      inferenceRegion,
      trainingRegion,
      statusText,
      statusTextRaw,
      errorMessage,
      jobId,
      jobStatus,
      inferenceLoading,
      trainingLoading,
      canRunInference,
      canTrain,
      statusClass,
      formatDate,
      loadReports,
      runInference,
      startTraining,
      openReport,
    }
  },
}
</script>
