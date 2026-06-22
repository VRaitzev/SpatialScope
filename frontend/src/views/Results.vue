<template>
  <div class="app-shell py-6">
    <div class="page-container space-y-6">
      <header class="glass-panel rounded-[2rem] px-6 py-5 lg:px-8">
        <div class="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <button class="ghost-btn" @click="$router.back()">Назад</button>
            <p class="mt-4 text-xs uppercase tracking-[0.28em] text-slate-400">SpatialScope · Report</p>
            <h1 class="mt-3 text-3xl font-semibold text-white">Результаты анализа</h1>
            <p class="mt-2 text-sm text-slate-400">Отчёт {{ reportId }} · Пресет {{ presetId || '—' }}</p>
          </div>
          <button class="secondary-btn" @click="exportJSON">Экспорт JSON</button>
        </div>
      </header>

      <div v-if="errorMessage" class="rounded-3xl border border-rose-400/20 bg-rose-400/10 px-5 py-4 text-sm text-rose-200">
        {{ errorMessage }}
      </div>

      <section class="grid grid-cols-1 gap-6 xl:grid-cols-[1.2fr_1fr]">
        <article class="glass-panel rounded-[2rem] p-6">
          <div class="flex items-center justify-between gap-3">
            <div>
              <h2 class="section-title">Предсказанная экспрессия генов</h2>
              <p class="section-subtitle mt-2 text-sm">Численные результаты пространственной модели для выбранного ROI.</p>
            </div>
            <span class="badge">{{ geneEntries.length }} genes</span>
          </div>

          <div v-if="geneEntries.length === 0" class="empty-state mt-6">
            <p class="text-lg font-medium text-white">Нет данных по экспрессии</p>
            <p class="mt-2 text-sm">В отчёте отсутствуют предсказанные значения генов.</p>
          </div>

          <div v-else class="mt-6 space-y-3">
            <div v-for="([gene, value]) in geneEntries" :key="gene" class="rounded-2xl border border-white/8 bg-white/5 p-4">
              <div class="flex items-center gap-4">
                <div class="w-20 shrink-0 text-sm font-semibold text-white">{{ gene }}</div>
                <div class="h-3 flex-1 overflow-hidden rounded-full bg-slate-800">
                  <div
                    class="h-full rounded-full bg-gradient-to-r from-cyan-400 via-violet-400 to-fuchsia-400"
                    :style="{ width: barWidth(value) }"
                  ></div>
                </div>
                <div class="w-16 shrink-0 text-right text-sm font-mono text-slate-300">{{ formatValue(value) }}</div>
              </div>
            </div>
          </div>
        </article>

        <article class="glass-panel rounded-[2rem] p-6">
          <div class="flex items-center justify-between gap-3">
            <div>
              <h2 class="section-title">Интерпретация моделей</h2>
              <p class="section-subtitle mt-2 text-sm">Пояснение вклада признаков и пространственных карт атрибуции.</p>
            </div>
            <span class="badge">{{ interpretations.length }} modules</span>
          </div>

          <div v-if="interpretations.length === 0" class="empty-state mt-6">
            <p class="text-lg font-medium text-white">Интерпретация недоступна</p>
            <p class="mt-2 text-sm">В выбранном отчёте нет данных пост-хок интерпретации.</p>
          </div>

          <div v-else class="mt-6 space-y-4">
            <div v-for="item in interpretations" :key="item.model_type" class="rounded-3xl border border-white/8 bg-white/5 p-5">
              <div class="flex items-start justify-between gap-4">
                <div>
                  <p class="text-base font-semibold text-white">{{ item.model_name }}</p>
                  <p class="mt-1 text-xs uppercase tracking-[0.18em] text-slate-500">{{ item.model_type }}</p>
                </div>
              </div>
              <p class="mt-3 text-sm leading-6 text-slate-300">{{ item.summary }}</p>

              <div v-if="item.top_features?.length" class="mt-4 space-y-2">
                <p class="text-xs uppercase tracking-[0.18em] text-slate-500">Top features</p>
                <div
                  v-for="feature in item.top_features.slice(0, 6)"
                  :key="`${item.model_type}-${feature.feature_name}`"
                  class="flex items-center justify-between rounded-2xl border border-white/8 bg-slate-950/40 px-4 py-3 text-sm"
                >
                  <span class="text-slate-300">{{ feature.feature_name }}</span>
                  <span class="font-mono text-white">{{ feature.importance.toFixed(3) }}</span>
                </div>
              </div>

              <div v-if="item.spatial_map?.values?.length" class="mt-5">
                <p class="mb-3 text-xs uppercase tracking-[0.18em] text-slate-500">Spatial attribution map</p>
                <canvas :ref="(el) => setCanvasRef(item.model_type, el)" class="w-full rounded-2xl border border-white/8 bg-slate-950/60"></canvas>
              </div>
            </div>
          </div>
        </article>
      </section>
    </div>
  </div>
</template>

<script>
import { computed, nextTick, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import api, { buildApiUrl } from '../api'

export default {
  name: 'ResultsView',
  setup() {
    const route = useRoute()
    const reportId = route.params.id
    const genes = ref({})
    const presetId = ref('')
    const interpretations = ref([])
    const errorMessage = ref('')
    const canvasRefs = {}

    const geneEntries = computed(() => Object.entries(genes.value || {}).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])))

    const setCanvasRef = (type, element) => {
      if (element) {
        canvasRefs[type] = element
      }
    }

    const barWidth = (value) => `${Math.min(Math.abs(Number(value) || 0) * 100, 100)}%`
    const formatValue = (value) => Number(value || 0).toFixed(3)

    const drawSpatialMap = (item) => {
      const canvas = canvasRefs[item.model_type]
      const values = item.spatial_map?.values
      if (!canvas || !values?.length) return

      const ctx = canvas.getContext('2d')
      const height = values.length
      const width = values[0].length
      canvas.width = width
      canvas.height = height
      const imageData = ctx.createImageData(width, height)

      for (let y = 0; y < height; y += 1) {
        for (let x = 0; x < width; x += 1) {
          const value = values[y][x]
          const index = (y * width + x) * 4
          const color = Math.floor(value * 255)
          imageData.data[index] = color
          imageData.data[index + 1] = Math.floor(color * 0.45)
          imageData.data[index + 2] = 255 - color
          imageData.data[index + 3] = 255
        }
      }

      ctx.putImageData(imageData, 0, 0)
    }

    const loadReport = async () => {
      errorMessage.value = ''
      try {
        const response = await api.get(`/reports/${reportId}`)
        const data = response.data.data
        presetId.value = data.preset_id || data.roi_coordinates?.preset_id || '—'

        const predicted = data.predicted_expression || {}
        if (predicted.predictions?.length > 0) {
          genes.value = predicted.predictions[0].predicted_expressions || {}
        }
        interpretations.value = predicted.interpretation || []

        await nextTick()
        interpretations.value.forEach(drawSpatialMap)
      } catch (error) {
        errorMessage.value = error.response?.data?.detail || 'Не удалось загрузить отчёт'
      }
    }

    const exportJSON = async () => {
      try {
        const response = await api.get(`/reports/${reportId}/export/json`)
        const blob = new Blob([JSON.stringify(response.data, null, 2)], { type: 'application/json' })
        const url = URL.createObjectURL(blob)
        const link = document.createElement('a')
        link.href = url
        link.download = `report-${reportId}.json`
        document.body.appendChild(link)
        link.click()
        link.remove()
        URL.revokeObjectURL(url)
      } catch (error) {
        errorMessage.value = error.response?.data?.detail || 'Не удалось экспортировать отчёт'
      }
    }

    onMounted(loadReport)

    return {
      reportId,
      presetId,
      interpretations,
      errorMessage,
      geneEntries,
      setCanvasRef,
      barWidth,
      formatValue,
      exportJSON,
    }
  },
}
</script>
