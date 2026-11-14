import {readFileSync} from 'node:fs'
import {createRequire} from 'node:module'

// Reuse the browser test dependency already declared by the frontend submodule.
const require = createRequire(new URL('../ComfyUI-Easy-Use-Frontend/package.json', import.meta.url))
const {test, expect} = require('@playwright/test')

test('v1 toast keeps untrusted messages as text and preserves safe icons and links', async ({page}) => {
    const source = readFileSync(new URL('../web_version/v1/js/common/toast.js', import.meta.url), 'utf8')
        .replace(/^import .*;\s*$/gm, '')
        .replace('export const toast = new Toast();', 'window.auditToast = new Toast();')
    await page.setContent('<html><body></body></html>')
    await page.addScriptTag({content: 'const sleep = async () => {}; const $t = value => value;\n' + source})
    const payload = '<img src="data:image/png,invalid" onerror="window.__audit_marker=1">'
    await page.evaluate(async (content) => {
        await window.auditToast.showToast({
            id: 'audit', icon: 'error', content, duration: 0,
            actions: [
                {label: 'unsafe', href: 'javascript:window.__audit_marker=2'},
                {label: 'safe', href: 'https://example.com/'},
            ],
        })
    }, payload)
    await expect(page.locator('body')).toContainText(payload)
    await expect(page.locator('img')).toHaveCount(0)
    await expect(page.locator('svg')).toHaveCount(1)
    await expect(page.locator('a')).toHaveCount(1)
    await expect(page.locator('a')).toHaveAttribute('href', 'https://example.com/')
    await expect(page.locator('a')).toHaveAttribute('rel', 'noopener noreferrer')
    expect(await page.evaluate(() => window.__audit_marker)).toBeUndefined()
})
