<template>
  <div class="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100">
    <div class="bg-white rounded-2xl shadow-2xl p-8 w-96">
      <div class="text-center mb-8">
        <div class="text-5xl mb-4">🧬</div>
        <h1 class="text-3xl font-bold text-gray-800">Xenium Platform</h1>
        <p class="text-gray-500">Пространственная транскриптомика</p>
      </div>

      <form @submit.prevent="login">
        <div class="mb-4">
          <label class="block text-gray-700 text-sm font-bold mb-2">Email</label>
          <input v-model="email" type="email" required
                 class="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                 placeholder="doctor@clinic.com">
        </div>

        <div class="mb-6">
          <label class="block text-gray-700 text-sm font-bold mb-2">Пароль</label>
          <input v-model="password" type="password" required
                 class="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                 placeholder="••••••••">
        </div>

        <button type="submit"
                class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg transition">
          🔐 Войти
        </button>

        <p class="mt-4 text-center text-sm text-gray-600">
          Нет аккаунта? <a href="#" class="text-blue-600">Зарегистрироваться</a>
        </p>
      </form>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import api from '../api'

export default {
  setup() {
    const router = useRouter()
    const email = ref('')
    const password = ref('')

    const login = async () => {
      try {
        const response = await api.post('/users/login', {
          email: email.value,
          password_hash: password.value
        })
        
        localStorage.setItem('token', response.data.token)
        localStorage.setItem('user', JSON.stringify(response.data.user))
        
        router.push('/')
      } catch (error) {
        alert('Ошибка входа: ' + error.response?.data?.detail || 'Неверные данные')
      }
    }

    return { email, password, login }
  }
}
</script>