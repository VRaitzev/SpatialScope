import { createRouter, createWebHistory } from 'vue-router'
import Login from '../views/Login.vue'
import Register from '../views/Register.vue'
import Dashboard from '../views/Dashboard.vue'
import SlideViewer from '../views/SlideViewer.vue'
import Results from '../views/Results.vue'

const routes = [
  { path: '/login', name: 'login', component: Login, meta: { public: true } },
  { path: '/register', name: 'register', component: Register, meta: { public: true } },
  { path: '/', name: 'dashboard', component: Dashboard },
  { path: '/slide/:id', name: 'slide', component: SlideViewer },
  { path: '/results/:id', name: 'results', component: Results },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

router.beforeEach((to, from, next) => {
  const token = localStorage.getItem('token')
  const isPublic = Boolean(to.meta.public)

  if (!isPublic && !token) {
    next({ name: 'login' })
    return
  }

  if (isPublic && token) {
    next({ name: 'dashboard' })
    return
  }

  next()
})

export default router
