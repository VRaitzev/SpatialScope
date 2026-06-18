<template>
  <div class="min-h-screen bg-gray-50">
    <!-- Навбар -->
    <nav class="bg-white shadow-lg">
      <div class="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
        <h1 class="text-2xl font-bold text-blue-600">🧬 Xenium Platform</h1>
        <div class="flex items-center gap-4">
          <span class="text-gray-600">👤 {{ user.name }}</span>
          <span class="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">{{ user.role }}</span>
          <button @click="logout" class="text-red-500 hover:text-red-700">Выйти</button>
        </div>
      </div>
    </nav>

    <!-- Контент -->
    <div class="max-w-7xl mx-auto px-4 py-8">
      <!-- Статистика -->
      <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div class="bg-white rounded-xl shadow p-6">
          <div class="text-3xl font-bold text-blue-600">{{ projects.length }}</div>
          <div class="text-gray-500">Проектов</div>
        </div>
        <div class="bg-white rounded-xl shadow p-6">
          <div class="text-3xl font-bold text-green-600">{{ totalSlides }}</div>
          <div class="text-gray-500">Слайдов</div>
        </div>
        <div class="bg-white rounded-xl shadow p-6">
          <div class="text-3xl font-bold text-purple-600">{{ totalReports }}</div>
          <div class="text-gray-500">Отчетов</div>
        </div>
        <div class="bg-white rounded-xl shadow p-6">
          <div class="text-sm font-bold text-orange-600">{{ activePreset }}</div>
          <div class="text-gray-500">Активная модель</div>
        </div>
      </div>

      <!-- Проекты -->
      <div class="bg-white rounded-xl shadow-lg p-6">
        <div class="flex justify-between items-center mb-6">
          <h2 class="text-xl font-bold">📁 Мои проекты</h2>
          <button @click="createProject" class="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700">
            ➕ Новый проект
          </button>
        </div>

        <div v-if="projects.length === 0" class="text-center py-8 text-gray-500">
          У вас пока нет проектов. Создайте первый!
        </div>

        <div v-for="project in projects" :key="project.project_id" 
             class="border rounded-lg p-4 mb-4 hover:shadow-md transition">
          <div class="flex justify-between items-start">
            <div>
              <h3 class="font-bold text-lg">{{ project.name }}</h3>
              <p class="text-gray-500 text-sm">{{ project.description || 'Без описания' }}</p>
              <div class="flex flex-wrap gap-2 mt-2">
                <span class="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs">
                  {{ project.slides_count }} слайдов
                </span>
                <span v-for="slide in project.slides" :key="slide.id" 
                      class="bg-gray-100 px-2 py-1 rounded text-xs">
                  {{ slide.patient }} {{ slide.status === 'ready' ? '✅' : slide.status === 'processing' ? '⏳' : '❌' }}
                </span>
              </div>
            </div>
            <button @click="openSlide(project.slides[0]?.id)" 
                    class="bg-gray-100 px-4 py-2 rounded hover:bg-gray-200">
              Открыть →
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import api from '../api'

export default {
  setup() {
    const router = useRouter()
    const user = ref(JSON.parse(localStorage.getItem('user') || '{"name":"Доктор","role":"clinician"}'))
    const projects = ref([])
    const activePreset = ref('Загрузка...')

    const totalSlides = computed(() => {
      return projects.value.reduce((sum, p) => sum + (p.slides?.length || 0), 0)
    })

    const totalReports = computed(() => {
      return projects.value.reduce((sum, p) => {
        return sum + (p.slides?.reduce((s, slide) => s + (slide.reports?.length || 0), 0) || 0)
      }, 0)
    })

    const loadData = async () => {
      try {
        const response = await api.get('/projects')
        projects.value = response.data.data.projects || []

        const presetResp = await api.get('/presets')
        const presets = presetResp.data.data || []
        const active = presets.find(p => p.is_active)
        activePreset.value = active ? active.name : 'Нет активной модели'
      } catch (e) {
        console.error('Ошибка загрузки:', e)
      }
    }

    const createProject = async () => {
      const name = prompt('Введите название проекта:')
      if (name) {
        try {
          await api.post('/projects', { name })
          await loadData()
        } catch (e) {
          alert('Ошибка создания проекта')
        }
      }
    }

    const openSlide = (slideId) => {
      if (slideId) {
        router.push(`/slide/${slideId}`)
      } else {
        alert('В проекте нет слайдов. Загрузите слайд!')
      }
    }

    const logout = () => {
      localStorage.removeItem('token')
      localStorage.removeItem('user')
      router.push('/login')
    }

    onMounted(loadData)

    return { user, projects, totalSlides, totalReports, activePreset, createProject, openSlide, logout }
  }
}
</script>