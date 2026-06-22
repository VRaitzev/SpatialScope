<template>
  <div class="app-shell py-6">
    <div class="page-container space-y-6">
      <header class="glass-panel rounded-[2rem] px-6 py-5 lg:px-8">
        <div class="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p class="text-xs uppercase tracking-[0.28em] text-slate-400">SpatialScope</p>
            <h1 class="mt-3 text-3xl font-semibold text-white">Операционная панель пространственного анализа</h1>
            <p class="mt-2 max-w-3xl text-sm leading-6 text-slate-400">
              Единый интерфейс для управления проектами, слайдами Xenium, модельными пресетами,
              обучением GNN и интерпретацией результатов для всех ролей платформы.
            </p>
          </div>

          <div class="flex flex-col gap-3 lg:items-end">
            <div class="flex flex-wrap items-center gap-3">
              <span class="badge" :class="roleBadgeClass">{{ roleLabel }}</span>
              <span class="badge">{{ user.email || 'unknown@spatialscope.local' }}</span>
            </div>
            <div class="text-right">
              <p class="text-lg font-medium text-white">{{ user.name || 'Пользователь' }}</p>
              <p class="text-sm text-slate-400">{{ roleDescription }}</p>
            </div>
            <button class="ghost-btn" @click="logout">Выйти</button>
          </div>
        </div>
      </header>

      <div v-if="globalError" class="rounded-3xl border border-rose-400/20 bg-rose-400/10 px-5 py-4 text-sm text-rose-200">
        {{ globalError }}
      </div>

      <section class="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        <article class="metric-card glass-panel-soft">
          <p class="metric-label">Проекты</p>
          <p class="metric-value mt-3 text-cyan-300">{{ projects.length }}</p>
          <p class="mt-2 text-sm text-slate-400">Активные пространства анализа</p>
        </article>
        <article class="metric-card glass-panel-soft">
          <p class="metric-label">Слайды</p>
          <p class="metric-value mt-3 text-emerald-300">{{ totalSlides }}</p>
          <p class="mt-2 text-sm text-slate-400">Зарегистрированные Xenium-слайды</p>
        </article>
        <article class="metric-card glass-panel-soft">
          <p class="metric-label">Отчёты</p>
          <p class="metric-value mt-3 text-fuchsia-300">{{ totalReports }}</p>
          <p class="mt-2 text-sm text-slate-400">Сохранённые результаты инференса</p>
        </article>
        <article class="metric-card glass-panel-soft">
          <p class="metric-label">Активный пресет</p>
          <p class="mt-3 text-lg font-semibold text-violet-300">{{ activePreset?.name || 'Не выбран' }}</p>
          <p class="mt-2 text-sm text-slate-400">{{ activePreset?.description || 'Пресет будет применяться по умолчанию' }}</p>
        </article>
      </section>

      <section class="grid grid-cols-1 gap-6 xl:grid-cols-[1.6fr_1fr]">
        <div class="space-y-6">
          <article class="glass-panel rounded-[2rem] p-6">
            <div class="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
              <div>
                <h2 class="section-title">Проекты и слайды</h2>
                <p class="section-subtitle mt-2 text-sm">Просмотр структуры проектов, слайдов и истории отчётов.</p>
              </div>
              <div class="flex flex-wrap gap-3">
                <button v-if="canManageProjects" class="secondary-btn" @click="toggleProjectForm">
                  {{ showProjectForm ? 'Скрыть форму проекта' : 'Новый проект' }}
                </button>
                <button v-if="canRegisterSlides" class="secondary-btn" @click="toggleSlideForm">
                  {{ showSlideForm ? 'Скрыть форму слайда' : 'Регистрация слайда' }}
                </button>
                <button class="ghost-btn" @click="loadData">Обновить данные</button>
              </div>
            </div>

            <div v-if="showProjectForm && canManageProjects" class="mt-6 grid gap-4 rounded-3xl border border-white/8 bg-white/5 p-5 lg:grid-cols-2">
              <div>
                <label class="mb-2 block text-sm font-medium text-slate-300">Название проекта</label>
                <input v-model="projectForm.name" class="input-control" type="text" placeholder="Например, Cohort A / Lung 2026">
              </div>
              <div>
                <label class="mb-2 block text-sm font-medium text-slate-300">Описание</label>
                <input v-model="projectForm.description" class="input-control" type="text" placeholder="Краткое описание проекта">
              </div>
              <div class="lg:col-span-2 flex justify-end">
                <button class="primary-btn" :disabled="projectSubmitting" @click="createProject">
                  {{ projectSubmitting ? 'Создание...' : 'Создать проект' }}
                </button>
              </div>
            </div>

            <div v-if="showSlideForm && canRegisterSlides" class="mt-6 grid gap-4 rounded-3xl border border-white/8 bg-white/5 p-5 lg:grid-cols-2">
              <div>
                <label class="mb-2 block text-sm font-medium text-slate-300">Проект</label>
                <select v-model="slideForm.project_id" class="select-control">
                  <option value="">Выберите проект</option>
                  <option v-for="project in projects" :key="project.project_id" :value="project.project_id">
                    {{ project.name }}
                  </option>
                </select>
              </div>
              <div>
                <label class="mb-2 block text-sm font-medium text-slate-300">Пациент / слайд</label>
                <input v-model="slideForm.patient_name" class="input-control" type="text" placeholder="Patient-001">
              </div>
              <div>
                <label class="mb-2 block text-sm font-medium text-slate-300">Путь к OME-TIFF</label>
                <input v-model="slideForm.tiff_path" class="input-control" type="text" placeholder="D:\\data\\sample.ome.tif">
              </div>
              <div>
                <label class="mb-2 block text-sm font-medium text-slate-300">Путь к Parquet</label>
                <input v-model="slideForm.parquet_path" class="input-control" type="text" placeholder="D:\\data\\cells.parquet">
              </div>
              <div class="lg:col-span-2">
                <label class="mb-2 block text-sm font-medium text-slate-300">Путь к HDF5</label>
                <input v-model="slideForm.hdf5_path" class="input-control" type="text" placeholder="D:\\data\\matrix.h5">
              </div>
              <div class="lg:col-span-2 flex justify-end">
                <button class="primary-btn" :disabled="slideSubmitting" @click="registerSlide">
                  {{ slideSubmitting ? 'Регистрация...' : 'Зарегистрировать слайд' }}
                </button>
              </div>
            </div>

            <div v-if="projects.length === 0" class="empty-state mt-6">
              <p class="text-lg font-medium text-white">Проекты ещё не созданы</p>
              <p class="mt-2 max-w-xl text-sm">Создайте проект, затем зарегистрируйте Xenium-слайд и переходите к анализу.</p>
            </div>

            <div v-else class="mt-6 space-y-4">
              <article v-for="project in projects" :key="project.project_id" class="rounded-3xl border border-white/8 bg-white/5 p-5">
                <div class="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                  <div>
                    <div class="flex flex-wrap items-center gap-3">
                      <h3 class="text-xl font-semibold text-white">{{ project.name }}</h3>
                      <span class="badge">{{ project.slides_count || project.slides?.length || 0 }} slides</span>
                    </div>
                    <p class="mt-2 text-sm text-slate-400">{{ project.description || 'Описание проекта не заполнено.' }}</p>
                  </div>
                  <div class="flex flex-wrap gap-3">
                    <button
                      class="secondary-btn"
                      :disabled="!project.slides?.length"
                      @click="openSlide(project.slides[0]?.id)"
                    >
                      Открыть первый слайд
                    </button>
                  </div>
                </div>

                <div class="mt-5 table-card">
                  <div class="table-grid">
                    <div class="table-row header">
                      <span>Слайд</span>
                      <span>Статус</span>
                      <span>Отчёты</span>
                      <span>Последнее действие</span>
                      <span>Действия</span>
                    </div>
                    <div v-for="slide in project.slides || []" :key="slide.id" class="table-row">
                      <div>
                        <p class="font-medium text-white">{{ slide.patient }}</p>
                        <p class="mt-1 text-xs text-slate-500">ID: {{ slide.id }}</p>
                      </div>
                      <div>
                        <span class="status-pill" :class="statusClass(slide.status)">{{ slide.status || 'unknown' }}</span>
                      </div>
                      <div>
                        <p class="text-white">{{ slide.reports?.length || 0 }}</p>
                      </div>
                      <div>
                        <p class="text-sm text-slate-400">{{ slide.reports?.[0]?.created_at ? formatDate(slide.reports[0].created_at) : 'Нет активности' }}</p>
                      </div>
                      <div class="flex flex-wrap gap-2">
                        <button class="ghost-btn" @click="openSlide(slide.id)">Просмотр</button>
                        <button
                          class="ghost-btn"
                          :disabled="!(slide.reports?.length)
                          "
                          @click="openReport(slide.reports?.[0]?.id)"
                        >
                          Последний отчёт
                        </button>
                      </div>
                    </div>
                    <div v-if="!(project.slides || []).length" class="px-5 py-6 text-sm text-slate-400">
                      В проекте пока нет зарегистрированных слайдов.
                    </div>
                  </div>
                </div>
              </article>
            </div>
          </article>
        </div>

        <div class="space-y-6">
          <article class="glass-panel rounded-[2rem] p-6">
            <h2 class="section-title">Ролевые возможности</h2>
            <p class="section-subtitle mt-2 text-sm">Интерфейс адаптируется под сценарии каждой роли.</p>
            <div class="mt-5 space-y-3 text-sm text-slate-300">
              <div class="rounded-2xl border border-white/8 bg-white/5 p-4">
                <p class="font-medium text-white">Administrator</p>
                <p class="mt-2">Управление пользователями и активацией пресетов, контроль состояния платформы.</p>
              </div>
              <div class="rounded-2xl border border-white/8 bg-white/5 p-4">
                <p class="font-medium text-white">Researcher</p>
                <p class="mt-2">Регистрация слайдов, обучение моделей, анализ данных и работа с ROI.</p>
              </div>
              <div class="rounded-2xl border border-white/8 bg-white/5 p-4">
                <p class="font-medium text-white">Clinician</p>
                <p class="mt-2">Просмотр слайдов, запуск инференса, изучение отчётов и интерпретации.</p>
              </div>
            </div>
          </article>

          <article class="glass-panel rounded-[2rem] p-6">
            <div class="flex items-start justify-between gap-4">
              <div>
                <h2 class="section-title">Пресеты моделей</h2>
                <p class="section-subtitle mt-2 text-sm">Доступные комбинации экспертов и пространственных моделей.</p>
              </div>
              <button v-if="canManagePresets" class="ghost-btn" @click="loadPresets">Обновить</button>
            </div>

            <div class="mt-5 space-y-3">
              <div v-for="preset in presets" :key="preset.id" class="rounded-2xl border border-white/8 bg-white/5 p-4">
                <div class="flex flex-col gap-3">
                  <div>
                    <div class="flex items-center gap-3">
                      <p class="font-medium text-white">{{ preset.name }}</p>
                      <span v-if="preset.is_active" class="status-pill status-success">active</span>
                    </div>
                    <p class="mt-2 text-sm text-slate-400">{{ preset.description }}</p>
                  </div>
                  <div class="text-xs text-slate-500">
                    Expert: {{ preset.expert_id }} · Spatial model: {{ preset.spatial_model_id }}
                  </div>
                  <button
                    v-if="canManagePresets"
                    class="secondary-btn"
                    :disabled="preset.is_active || presetActivatingId === preset.id"
                    @click="activatePreset(preset.id)"
                  >
                    {{ presetActivatingId === preset.id ? 'Активация...' : 'Сделать активным' }}
                  </button>
                </div>
              </div>
              <div v-if="presets.length === 0" class="text-sm text-slate-400">Нет зарегистрированных пресетов.</div>
            </div>
          </article>

          <article class="glass-panel rounded-[2rem] p-6">
            <h2 class="section-title">Сводка доступа</h2>
            <div class="mt-4 space-y-2 text-sm text-slate-300">
              <div class="flex items-center justify-between rounded-2xl border border-white/8 bg-white/5 px-4 py-3">
                <span>Создание проектов</span>
                <span class="badge">{{ canManageProjects ? 'Доступно' : 'Нет доступа' }}</span>
              </div>
              <div class="flex items-center justify-between rounded-2xl border border-white/8 bg-white/5 px-4 py-3">
                <span>Регистрация слайдов</span>
                <span class="badge">{{ canRegisterSlides ? 'Доступно' : 'Нет доступа' }}</span>
              </div>
              <div class="flex items-center justify-between rounded-2xl border border-white/8 bg-white/5 px-4 py-3">
                <span>Активация пресетов</span>
                <span class="badge">{{ canManagePresets ? 'Доступно' : 'Нет доступа' }}</span>
              </div>
              <div class="flex items-center justify-between rounded-2xl border border-white/8 bg-white/5 px-4 py-3">
                <span>Инференс и отчёты</span>
                <span class="badge">{{ canViewReports ? 'Доступно' : 'Нет доступа' }}</span>
              </div>
            </div>
          </article>
        </div>
      </section>
    </div>
  </div>
</template>

<script>
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import api from '../api'

const ROLE_LABELS = {
  admin: 'Administrator',
  researcher: 'Researcher',
  clinician: 'Clinician',
}

const ROLE_DESCRIPTIONS = {
  admin: 'Полный контроль доступа, пользователей и модельных конфигураций.',
  researcher: 'Подготовка данных, регистрация слайдов и обучение моделей.',
  clinician: 'Проведение анализа, просмотр слайдов и интерпретация отчётов.',
}

export default {
  name: 'DashboardView',
  setup() {
    const router = useRouter()
    const user = ref(JSON.parse(localStorage.getItem('user') || '{"name":"User","role":"clinician"}'))
    const projects = ref([])
    const presets = ref([])
    const globalError = ref('')
    const showProjectForm = ref(false)
    const showSlideForm = ref(false)
    const projectSubmitting = ref(false)
    const slideSubmitting = ref(false)
    const presetActivatingId = ref('')

    const projectForm = ref({
      name: '',
      description: '',
    })

    const slideForm = ref({
      project_id: '',
      patient_name: '',
      tiff_path: '',
      parquet_path: '',
      hdf5_path: '',
    })

    const role = computed(() => user.value?.role || 'clinician')
    const roleLabel = computed(() => ROLE_LABELS[role.value] || role.value)
    const roleDescription = computed(() => ROLE_DESCRIPTIONS[role.value] || 'Роль пользователя не определена')
    const roleBadgeClass = computed(() => `badge-role-${role.value}`)

    const canManageProjects = computed(() => ['admin', 'researcher'].includes(role.value))
    const canRegisterSlides = computed(() => ['admin', 'researcher'].includes(role.value))
    const canManagePresets = computed(() => role.value === 'admin')
    const canViewReports = computed(() => ['admin', 'researcher', 'clinician'].includes(role.value))

    const totalSlides = computed(() => projects.value.reduce((sum, project) => sum + (project.slides?.length || 0), 0))
    const totalReports = computed(() => projects.value.reduce((sum, project) => {
      return sum + (project.slides?.reduce((slideSum, slide) => slideSum + (slide.reports?.length || 0), 0) || 0)
    }, 0))

    const activePreset = computed(() => presets.value.find((preset) => preset.is_active) || null)

    const formatDate = (value) => {
      if (!value) return '—'
      return new Date(value).toLocaleString('ru-RU')
    }

    const statusClass = (status) => `status-${status || 'queued'}`

    const normalizeProjects = (payload) => {
      const rawProjects = payload?.projects || []
      return rawProjects.map((project) => ({
        ...project,
        slides: (project.slides || []).map((slide) => ({
          ...slide,
          reports: slide.reports || [],
        })),
      }))
    }

    const loadProjects = async () => {
      const response = await api.get('/projects')
      projects.value = normalizeProjects(response.data.data)
      if (!slideForm.value.project_id && projects.value.length) {
        slideForm.value.project_id = projects.value[0].project_id
      }
    }

    const loadPresets = async () => {
      const response = await api.get('/presets')
      presets.value = response.data.data || []
    }

    const loadData = async () => {
      globalError.value = ''
      try {
        await Promise.all([loadProjects(), loadPresets()])
      } catch (error) {
        globalError.value = error.response?.data?.detail || 'Не удалось загрузить данные панели управления'
      }
    }

    const resetProjectForm = () => {
      projectForm.value = { name: '', description: '' }
    }

    const resetSlideForm = () => {
      slideForm.value = {
        project_id: projects.value[0]?.project_id || '',
        patient_name: '',
        tiff_path: '',
        parquet_path: '',
        hdf5_path: '',
      }
    }

    const toggleProjectForm = () => {
      showProjectForm.value = !showProjectForm.value
      if (showProjectForm.value) {
        showSlideForm.value = false
      }
    }

    const toggleSlideForm = () => {
      showSlideForm.value = !showSlideForm.value
      if (showSlideForm.value) {
        showProjectForm.value = false
      }
    }

    const createProject = async () => {
      if (!projectForm.value.name.trim()) {
        globalError.value = 'Укажите название проекта'
        return
      }

      projectSubmitting.value = true
      globalError.value = ''
      try {
        await api.post('/projects', {
          name: projectForm.value.name,
          description: projectForm.value.description || null,
        })
        resetProjectForm()
        showProjectForm.value = false
        await loadProjects()
      } catch (error) {
        globalError.value = error.response?.data?.detail || 'Не удалось создать проект'
      } finally {
        projectSubmitting.value = false
      }
    }

    const registerSlide = async () => {
      if (!slideForm.value.project_id || !slideForm.value.patient_name || !slideForm.value.tiff_path || !slideForm.value.parquet_path || !slideForm.value.hdf5_path) {
        globalError.value = 'Заполните все поля формы регистрации слайда'
        return
      }

      slideSubmitting.value = true
      globalError.value = ''
      try {
        await api.post('/slides/register', { ...slideForm.value })
        resetSlideForm()
        showSlideForm.value = false
        await loadProjects()
      } catch (error) {
        globalError.value = error.response?.data?.detail || 'Не удалось зарегистрировать слайд'
      } finally {
        slideSubmitting.value = false
      }
    }

    const activatePreset = async (presetId) => {
      presetActivatingId.value = presetId
      globalError.value = ''
      try {
        await api.post('/presets/activate', { preset_id: presetId })
        await loadPresets()
      } catch (error) {
        globalError.value = error.response?.data?.detail || 'Не удалось активировать пресет'
      } finally {
        presetActivatingId.value = ''
      }
    }

    const openSlide = (slideId) => {
      if (!slideId) {
        globalError.value = 'У выбранного проекта нет доступных слайдов'
        return
      }
      router.push(`/slide/${slideId}`)
    }

    const openReport = (reportId) => {
      if (!reportId) {
        globalError.value = 'Для этого слайда пока нет отчётов'
        return
      }
      router.push(`/results/${reportId}`)
    }

    const logout = () => {
      localStorage.removeItem('token')
      localStorage.removeItem('user')
      router.push('/login')
    }

    onMounted(async () => {
      resetSlideForm()
      await loadData()
    })

    return {
      user,
      projects,
      presets,
      activePreset,
      roleLabel,
      roleDescription,
      roleBadgeClass,
      canManageProjects,
      canRegisterSlides,
      canManagePresets,
      canViewReports,
      totalSlides,
      totalReports,
      globalError,
      showProjectForm,
      showSlideForm,
      projectForm,
      slideForm,
      projectSubmitting,
      slideSubmitting,
      presetActivatingId,
      formatDate,
      statusClass,
      toggleProjectForm,
      toggleSlideForm,
      loadData,
      loadPresets,
      createProject,
      registerSlide,
      activatePreset,
      openSlide,
      openReport,
      logout,
    }
  },
}
</script>
