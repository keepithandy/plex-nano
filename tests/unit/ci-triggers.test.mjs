import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

test('packaged prompt changes trigger maintenance CI for pull requests and master pushes', async () => {
  const workflow = await readFile(new URL('../../.github/workflows/typescript-maintenance.yml', import.meta.url), 'utf8');
  const lines = workflow.split(/\r?\n/);
  for (const event of ['pull_request', 'push']) {
    const start = lines.indexOf(`  ${event}:`);
    assert.ok(start >= 0, `${event} trigger must exist`);
    const end = lines.findIndex((line, index) => index > start && /^(?:  \w+:|\S)/.test(line));
    const block = lines.slice(start + 1, end < 0 ? undefined : end);
    assert.ok(block.includes('    paths:'), `${event} must have path filters`);
    assert.ok(block.includes('      - "prompts/**"'), `${event} must check packaged prompt changes`);
    if (event === 'push') {
      assert.ok(block.includes('    branches:') && block.includes('      - master'), 'push verification must cover master');
    }
  }
});
