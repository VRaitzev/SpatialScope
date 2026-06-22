<template>
  <div class="auth-layout app-shell">
    <section class="auth-brand">
      <div>
        <p class="text-sm uppercase tracking-[0.3em] text-cyan-300/70">SpatialScope</p>
        <h1 class="mt-6 text-5xl font-semibold leading-tight text-white">
          Unified workspace for spatial transcriptomics workflows
        </h1>
        <p class="mt-6 max-w-md text-base leading-7 text-slate-300">
          Анализируйте слайды, запускайте инференс, обучайте пространственные модели и
          управляйте проектами в едином защищённом интерфейсе.
        </p>
      </div>

      <div class="grid gap-4">
        <div class="glass-panel rounded-3xl p-5">
          <p class="text-sm text-slate-300">Доступные роли</p>
          <div class="mt-4 flex flex-wrap gap-2">
            <span class="badge badge-role-admin">Administrator</span>
            <span class="badge badge-role-researcher">Researcher</span>
            <span class="badge badge-role-clinician">Clinician</span>
          </div>
        </div>
        <div class="glass-panel-soft rounded-3xl p-5">
          <p class="text-sm text-slate-400">Ключевые возможности</p>
          <ul class="mt-3 space-y-2 text-sm text-slate-300">
            <li>Управление проектами и Xenium-слайдами</li>
            <li>Инференс по ROI с интерпретацией моделей</li>
            <li>Управление пресетами и обучением GNN</li>
          </ul>
        </div>
      </div>
    </section>

    <section class="flex items-center justify-center p-6 lg:p-10">
      <div class="auth-card glass-panel neon-ring">
        <div>
          <p class="text-sm uppercase tracking-[0.25em] text-slate-400">SpatialScope</p>
          <h2 class="mt-4 text-3xl font-semibold text-white">Вход в систему</h2>
          <p class="mt-2 text-sm text-slate-400">Введите учётные данные для доступа к рабочему пространству.</p>
        </div>

        <form class="mt-8 space-y-5" @submit.prevent="login">
          <div>
            <label class="mb-2 block text-sm font-medium text-slate-300">Email</label>
            <input
              v-model="email"
              type="email"
              required
              class="input-control"
              placeholder="doctor@spatialscope.local"
            >
          </div>

          <div>
            <label class="mb-2 block text-sm font-medium text-slate-300">Пароль</label>
            <input
              v-model="password"
              type="password"
              required
              class="input-control"
              placeholder="Введите пароль"
            >
          </div>

          <div v-if="errorMessage" class="rounded-2xl border border-rose-400/20 bg-rose-400/10 px-4 py-3 text-sm text-rose-200">
            {{ errorMessage }}
          </div>

          <button type="submit" class="primary-btn w-full" :disabled="isSubmitting">
            {{ isSubmitting ? 'Выполняется вход...' : 'Войти' }}
          </button>

          <p class="text-center text-sm text-slate-400">
            Нет аккаунта?
            <router-link to="/register" class="font-medium text-cyan-300 hover:text-cyan-200">
              Зарегистрироваться
            </router-link>
          </p>
        </form>
      </div>
    </section>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import api from '../api'

export default {
  name: 'LoginView',
  setup() {
    const router = useRouter()
    const email = ref('')
    const password = ref('')
    const isSubmitting = ref(false)
    const errorMessage = ref('')

    const login = async () => {
      isSubmitting.value = true
      errorMessage.value = ''

      try {
        const response = await api.post('/users/login', {
          email: email.value,
          password: password.value,
        })

        localStorage.setItem('token', response.data.token)
        localStorage.setItem('user', JSON.stringify(response.data.user))
        router.push('/')
      } catch (error) {
        errorMessage.value = error.response?.data?.detail || 'Не удалось выполнить вход'
      } finally {
        isSubmitting.value = false
      }
    }

    return {
      email,
      password,
      isSubmitting,
      errorMessage,
      login,
    }
  },
}
</script>
