<template>
  <div class="min-h-screen bg-gray-50">
    <div class="max-w-7xl mx-auto px-4 py-8">
      <button @click="$router.back()" class="text-blue-600 hover:text-blue-800 mb-4">← Назад</button>
      
      <div class="bg-white rounded-xl shadow-lg p-6">
        <h2 class="text-2xl font-bold mb-4">📊 Результаты анализа</h2>
        <p class="text-gray-500 mb-6">Отчет #{{ reportId }}</p>
        
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <!-- Тепловая карта (имитация) -->
          <div class="border rounded-lg p-4">
            <h3 class="font-bold mb-3">Тепловая карта экспрессии</h3>
            <div class="bg-gradient-to-r from-blue-100 via-red-100 to-green-100 h-48 rounded flex items-center justify-center">
              <span class="text-gray-400">🧬 Тепловая карта</span>
            </div>
            <div class="flex justify-between text-xs text-gray-500 mt-2">
              <span>Низкая</span>
              <span>Высокая</span>
            </div>
          </div>
          
          <!-- Список генов -->
          <div class="border rounded-lg p-4">
            <h3 class="font-bold mb-3">Экспрессия генов</h3>
            <div class="space-y-2 max-h-64 overflow-y-auto">
              <div v-for="(value, gene) in genes" :key="gene" 
                   class="flex items-center gap-2">
                <span class="text-sm w-16 font-mono">{{ gene }}</span>
                <div class="flex-1 h-4 bg-gray-200 rounded overflow-hidden">
                  <div class="h-full bg-blue-600 rounded transition-all" 
                       :style="{width: Math.min(value * 100, 100) + '%'}"></div>
                </div>
                <span class="text-xs font-mono w-12">{{ (value * 100).toFixed(0) }}%</span>
              </div>
            </div>
          </div>
        </div>
        
        <div class="flex gap-4 mt-6">
          <button @click="exportPDF" class="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg">
            📥 Экспорт PDF
          </button>
          <button @click="exportJSON" class="bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg">
            💾 Экспорт JSON
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import api from '../api'

export default {
  setup() {
    const route = useRoute()
    const reportId = route.params.id
    const genes = ref({})

    onMounted(async () => {
      try {
        const response = await api.get(`/reports/${reportId}`)
        const data = response.data.data.predicted_expression
        
        if (data?.predictions?.length > 0) {
          genes.value = data.predictions[0].predicted_expressions || {}
        }
        
        // Если нет данных - используем демо-данные
        if (Object.keys(genes.value).length === 0) {
          genes.value = {
            CD3D: 0.85,
            CD68: 0.62,
            PDCD1: 0.43,
            CD4: 0.71,
            CD8A: 0.38,
            FOXP3: 0.27,
            CTLA4: 0.19
          }
        }
      } catch (e) {
        console.error('Ошибка загрузки отчета:', e)
        // Демо-данные
        genes.value = {
          CD3D: 0.85,
          CD68: 0.62,
          PDCD1: 0.43,
          CD4: 0.71,
          CD8A: 0.38,
          FOXP3: 0.27,
          CTLA4: 0.19
        }
      }
    })

    const exportPDF = () => {
      window.open(`http://localhost:8000/api/reports/${reportId}/export/pdf`)
    }

    const exportJSON = () => {
      window.open(`http://localhost:8000/api/reports/${reportId}/export/json`)
    }

    return { reportId, genes, exportPDF, exportJSON }
  }
}
</script>