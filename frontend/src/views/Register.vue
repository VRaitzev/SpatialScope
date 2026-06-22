<template>
  <div class="auth-layout app-shell">
    <section class="auth-brand">
      <div>
        <p class="text-sm uppercase tracking-[0.3em] text-violet-300/70">SpatialScope</p>
        <h1 class="mt-6 text-5xl font-semibold leading-tight text-white">
          Configure role-based access to the platform
        </h1>
        <p class="mt-6 max-w-md text-base leading-7 text-slate-300">
          Создавайте учётные записи для врачей, исследователей и администраторов с
          разграничением доступных сценариев работы в системе.
        </p>
      </div>

      <div class="glass-panel rounded-3xl p-6">
        <p class="text-sm text-slate-400">Ролевые области ответственности</p>
        <div class="mt-4 grid gap-4 text-sm text-slate-300">
          <div>
            <p class="font-medium text-white">Administrator</p>
            <p>Управление доступом, пользователями и модельными пресетами.</p>
          </div>
          <div>
            <p class="font-medium text-white">Researcher</p>
            <p>Регистрация слайдов, обучение моделей и исследовательский анализ.</p>
          </div>
          <div>
            <p class="font-medium text-white">Clinician</p>
            <p>Просмотр слайдов, запуск инференса и анализ отчётов.</p>
          </div>
        </div>
      </div>
    </section>

    <section class="flex items-center justify-center p-6 lg:p-10">
      <div class="auth-card glass-panel neon-ring">
        <div>
          <p class="text-sm uppercase tracking-[0.25em] text-slate-400">SpatialScope</p>
          <h2 class="mt-4 text-3xl font-semibold text-white">Регистрация пользователя</h2>
          <p class="mt-2 text-sm text-slate-400">Создайте новую учётную запись для работы с платформой.</p>
        </div>

        <form class="mt-8 space-y-5" @submit.prevent="register">
          <div>
            <label class="mb-2 block text-sm font-medium text-slate-300">Имя</label>
            <input v-model="name" type="text" required class="input-control" placeholder="Иван Петров">
          </div>

          <div>
            <label class="mb-2 block text-sm font-medium text-slate-300">Email</label>
            <input v-model="email" type="email" required class="input-control" placeholder="user@spatialscope.local">
          </div>

          <div>
            <label class="mb-2 block text-sm font-medium text-slate-300">Пароль</label>
            <input v-model="password" type="password" required class="input-control" placeholder="Минимум 6 символов">
          </div>

          <div>
            <label class="mb-2 block text-sm font-medium text-slate-300">Роль</label>
            <select v-model="role" class="select-control">
              <option value="clinician">Clinician</option>
              <option value="researcher">Researcher</option>
              <option value="admin">Administrator</option>
            </select>
          </div>

          <div v-if="successMessage" class="rounded-2xl border border-emerald-400/20 bg-emerald-400/10 px-4 py-3 text-sm text-emerald-200">
            {{ successMessage }}
          </div>

          <div v-if="errorMessage" class="rounded-2xl border border-rose-400/20 bg-rose-400/10 px-4 py-3 text-sm text-rose-200">
            {{ errorMessage }}
          </div>

          <button type="submit" class="primary-btn w-full" :disabled="isSubmitting">
            {{ isSubmitting ? 'Создание аккаунта...' : 'Зарегистрировать пользователя' }}
          </button>

          <p class="text-center text-sm text-slate-400">
            Уже есть аккаунт?
            <router-link to="/login" class="font-medium text-cyan-300 hover:text-cyan-200">Войти</router-link>
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
  name: 'RegisterView',
  setup() {
    const router = useRouter()
    const name = ref('')
    const email = ref('')
    const password = ref('')
    const role = ref('clinician')
    const isSubmitting = ref(false)
    const errorMessage = ref('')
    const successMessage = ref('')

    const register = async () => {
      isSubmitting.value = true
      errorMessage.value = ''
      successMessage.value = ''

      try {
        const response = await api.post('/users/register', {
          name: name.value,
          email: email.value,
          password: password.value,
          role: role.value,
        })

        if (response.data.success) {
          successMessage.value = 'Пользователь успешно зарегистрирован. Теперь можно выполнить вход.'
          setTimeout(() => router.push('/login'), 900)
        }
      } catch (error) {
        errorMessage.value = error.response?.data?.detail || 'Не удалось зарегистрировать пользователя'
      } finally {
        isSubmitting.value = false
      }
    }

    return {
      name,
      email,
      password,
      role,
      isSubmitting,
      errorMessage,
      successMessage,
      register,
    }
  },
}
</script>
