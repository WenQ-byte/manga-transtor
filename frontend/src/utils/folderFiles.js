const IMAGE_EXTENSIONS = new Set(['jpg', 'jpeg', 'png', 'webp', 'bmp'])

export function isImageFile(file) {
  const name = file?.name || ''
  return IMAGE_EXTENSIONS.has(name.split('.').pop()?.toLowerCase() || '')
}

function naturalPath(file) {
  return (file?.webkitRelativePath || file?.relativePath || file?.name || '').replaceAll('\\', '/')
}

function naturalCompare(left, right) {
  const a = naturalPath(left).toLocaleLowerCase()
  const b = naturalPath(right).toLocaleLowerCase()
  const aParts = a.split(/(\d+)/)
  const bParts = b.split(/(\d+)/)
  const length = Math.max(aParts.length, bParts.length)
  for (let index = 0; index < length; index += 1) {
    const aPart = aParts[index] || ''
    const bPart = bParts[index] || ''
    if (/^\d+$/.test(aPart) && /^\d+$/.test(bPart)) {
      const difference = Number(aPart) - Number(bPart)
      if (difference) return difference
    } else if (aPart !== bPart) {
      return aPart.localeCompare(bPart, undefined, { numeric: true })
    }
  }
  return naturalPath(left).localeCompare(naturalPath(right))
}

export function sortImageFiles(files) {
  return [...files].sort(naturalCompare)
}

function readEntryFile(entry) {
  return new Promise((resolve, reject) => entry.file(resolve, reject))
}

async function readDirectory(entry, rootPath = '') {
  const files = []
  const reader = entry.createReader()
  let entries
  do {
    entries = await new Promise((resolve, reject) => reader.readEntries(resolve, reject))
    for (const child of entries) {
      const childPath = rootPath ? `${rootPath}/${child.name}` : child.name
      if (child.isDirectory) files.push(...await readDirectory(child, childPath))
      else if (child.isFile) {
        try {
          const file = await readEntryFile(child)
          if (isImageFile(file)) {
            try { Object.defineProperty(file, 'relativePath', { value: childPath, configurable: true }) } catch { /* 某些浏览器的 File 对象不可扩展。 */ }
            files.push(file)
          }
        } catch {
          // 单个文件读取失败由调用方汇总提示，其他文件继续导入。
        }
      }
    }
  } while (entries.length)
  return files
}

export async function collectDroppedImageFiles(dataTransfer) {
  const items = Array.from(dataTransfer?.items || [])
  const entries = items.map((item) => item.webkitGetAsEntry?.()).filter(Boolean)
  const hasDirectory = entries.some((entry) => entry.isDirectory)
  if (!hasDirectory) {
    const files = Array.from(dataTransfer?.files || [])
    return { files: files.some((file) => file.webkitRelativePath) ? sortImageFiles(files) : files, hasDirectory: false }
  }

  const files = []
  for (const entry of entries) {
    if (entry.isDirectory) files.push(...await readDirectory(entry))
    else if (entry.isFile) {
      try {
        const file = await readEntryFile(entry)
        if (isImageFile(file)) files.push(file)
      } catch {
        // 继续处理剩余文件。
      }
    }
  }
  return { files: sortImageFiles(files), hasDirectory: true }
}
