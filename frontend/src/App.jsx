import { ToastProvider } from './components/Toast'
import TranslatePanel from './components/TranslatePanel'
import ErrorBoundary from './components/ErrorBoundary'

export default function App() {
  return (
    <ErrorBoundary>
      <ToastProvider>
        <main className="relative min-h-screen">
          <TranslatePanel />
        </main>
      </ToastProvider>
    </ErrorBoundary>
  )
}
