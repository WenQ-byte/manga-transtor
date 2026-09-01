import { ToastProvider } from './components/Toast'
import TranslatePanel from './components/TranslatePanel'

export default function App() {
  return (
    <ToastProvider>
      <main className="relative min-h-screen">
        <TranslatePanel />
      </main>
    </ToastProvider>
  )
}
