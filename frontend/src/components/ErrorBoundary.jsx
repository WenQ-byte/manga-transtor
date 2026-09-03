import { Component } from 'react'

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { error: null }
  }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, info) {
    console.error('ErrorBoundary caught:', error, info)
  }

  render() {
    if (this.state.error) {
      return (
        <div className="fixed inset-0 z-[999] flex items-center justify-center bg-black/80 p-6">
          <div className="max-w-lg rounded-2xl border border-red-400/40 bg-red-500/10 p-6 text-left text-sm text-red-100">
            <p className="mb-2 text-base font-semibold text-red-200">界面发生错误</p>
            <pre className="whitespace-pre-wrap break-words text-xs leading-5 text-red-200/90">{String(this.state.error?.message || this.state.error)}</pre>
            <button
              type="button"
              onClick={() => this.setState({ error: null })}
              className="mt-4 rounded-lg bg-red-500 px-4 py-2 text-sm text-white hover:bg-red-400"
            >
              继续使用
            </button>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}
