import { createRouter, createWebHistory } from 'vue-router'
import Login from '../views/Login.vue'
import Dashboard from '../views/Dashboard.vue'
import SlideViewer from '../views/SlideViewer.vue'
import Results from '../views/Results.vue'

const routes = [
  { path: '/login', component: Login },
  { path: '/', component: Dashboard },
  { path: '/slide/:id', component: SlideViewer },
  { path: '/results/:id', component: Results }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

// Защита маршрутов
router.beforeEach((to, from, next) => {
  const token = localStorage.getItem('token')
  if (to.path !== '/login' && !token) {
    next('/login')
  } else {
    next()
  }
})

export default router