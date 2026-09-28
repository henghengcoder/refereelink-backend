import assert from 'node:assert/strict';
import test from 'node:test';
import { selectPitchPlayers, useDashboardStore } from '../src/store/dashboardStore.ts';

test('Pitch players selector reuses its empty snapshot', () => {
  const state = useDashboardStore.getState();

  assert.equal(state.frameState, null);
  assert.strictEqual(selectPitchPlayers(state), selectPitchPlayers(state));
  assert.deepEqual(selectPitchPlayers(state), []);
});
