<template>
  <div class="min-h-screen bg-gray-100">
    <!-- Навбар -->
    <div class="bg-white shadow">
      <div class="max-w-7xl mx-auto px-4 py-3 flex justify-between items-center">
        <button @click="$router.back()" class="text-blue-600 hover:text-blue-800">← Назад</button>
        <h2 class="font-bold">🧬 {{ slideName || 'Слайд' }}</h2>
        <span :class="statusClass">{{ statusText }}</span>
      </div>
    </div>

    <!-- WSI Просмотрщик -->
    <div class="max-w-7xl mx-auto px-4 py-4">
      <div class="bg-white rounded-xl shadow-lg overflow-hidden">
        <div id="wsi-viewer" style="width:100%;height:70vh;"></div>
        
        <!-- Панель управления -->
        <div class="p-4 border-t grid grid-cols-1 md:grid-cols-4 gap-4">
          <div class="flex items-center gap-2">
            <label class="text-sm">X:</label>
            <input v-model.number="roi.x" type="number" class="border rounded px-2 py-1 w-20 text-sm">
          </div>
          <div class="flex items-center gap-2">
            <label class="text-sm">Y:</label>
            <input v-model.number="roi.y" type="number" class="border rounded px-2 py-1 w-20 text-sm">
          </div>
          <div class="flex items-center gap-2">
            <label class="text-sm">Размер:</label>
            <input v-model.number="roi.size" type="number" class="border rounded px-2 py-1 w-20 text-sm">
          </div>
          <button @click="runInference" 
                  class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm">
            🧬 Анализировать
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, onBeforeUnmount } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import OpenSeadragon from 'openseadragon'
import api from '../api'

export default {
  setup() {
    const route = useRoute()
    const router = useRouter()
    const slideId = route.params.id
    const slideName = ref('')
    const viewer = ref(null)
    const roi = ref({ x: 0, y: 0, size: 512 })
    const statusText = ref('Загрузка...')
    const statusClass = ref('text-gray-500')

    onMounted(async () => {
      try {
        // Загружаем данные слайда
        const response = await api.get(`/slides/${slideId}`)
        slideName.value = response.data.data.patient || 'Слайд'
        
        statusText.value = 'Готов'
        statusClass.value = 'text-green-500'

        // Инициализируем OpenSeadragon
        viewer.value = OpenSeadragon({
          id: 'wsi-viewer',
          tileSources: {
            type: 'image',
            url: `http://localhost:8000/api/slides/${slideId}/image`
          },
          showNavigator: true,
          defaultZoomLevel: 1,
          minZoomLevel: 0.5,
          maxZoomLevel: 10
        })
      } catch (e) {
        statusText.value = 'Ошибка загрузки'
        statusClass.value = 'text-red-500'
        console.error(e)
      }
    })

    onBeforeUnmount(() => {
      if (viewer.value) {
        viewer.value.destroy()
      }
    })

    const runInference = async () => {
      statusText.value = '⏳ Выполняется...'
      statusClass.value = 'text-yellow-500'
      
      try {
        const response = await api.post('/inference/run', {
          slide_id: slideId,
          x: roi.value.x,
          y: roi.value.y,
          patch_size: roi.value.size,
          regional_spots: []
        })
        
        const reportId = response.data.data.report_id
        statusText.value = '✅ Готово!'
        statusClass.value = 'text-green-500'
        
        router.push(`/results/${reportId}`)
      } catch (e) {
        statusText.value = '❌ Ошибка'
        statusClass.value = 'text-red-500'
        alert('Ошибка анализа: ' + (e.response?.data?.detail || e.message))
      }
    }

    return { slideId, slideName, viewer, roi, statusText, statusClass, runInference }
  }
}
</script>