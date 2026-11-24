import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import { rayDirectorWithVsr, labelVsrUnverified } from '../web/director_vsr_cache_helpers.mjs';

function fixture() {
    const nodes = [
        { id: 1, type: 'RayMMH3DirectorModelAdapter', inputs: [] },
        { id: 2, type: 'SolAttnV100', inputs: [{ name: 'model', type: 'MODEL', link: 10 }] },
        { id: 3, type: 'MiniMaxH3Director', inputs: [
            { name: 'model', type: 'MODEL', link: 11 }, { name: 'vsr', link: 12 },
            { name: 'refine', link: 13 }] },
        { id: 4, type: 'MiniMaxH3DirectorRefine', inputs: [] },
    ];
    const graph = { _nodes: nodes, links: {
        10: { origin_id: 1 }, 11: { origin_id: 2 }, 12: { origin_id: 5 }, 13: { origin_id: 4 },
    }, getNodeById(id) { return nodes.find(n => String(n.id) === String(id)); } };
    return { nodes, graph };
}

test('only a Ray-backed Director with connected VSR is relabelled', () => {
    const { nodes, graph } = fixture();
    assert.equal(rayDirectorWithVsr(graph, nodes[2]), true);
    nodes[2].inputs[1].link = null;
    assert.equal(rayDirectorWithVsr(graph, nodes[2]), false);
    nodes[2].inputs[1].link = 12;
    nodes[0].type = 'UNETLoader';
    assert.equal(rayDirectorWithVsr(graph, nodes[2]), false);
});

test('model link cycles do not hang status rendering', () => {
    const { nodes, graph } = fixture();
    graph.links[10].origin_id = 2;
    assert.equal(rayDirectorWithVsr(graph, nodes[2]), false);
});

test('retain first-pass status but never assert verified VSR final matches', () => {
    const original = '一采匹配：是（3/3 段）\n成片匹配：3/3 段\n可以二采';
    const text = labelVsrUnverified(original);
    assert.match(text, /一采匹配：是（3\/3 段）/);
    assert.match(text, /成片匹配：未验证/);
    assert.ok(!text.includes('成片匹配：3/3'));
    assert.equal(labelVsrUnverified(text), text);
    assert.equal(labelVsrUnverified('正在检查分段缓存…'), '正在检查分段缓存…');
});

async function extension() {
    let registered;
    const helpers = await import('../web/director_vsr_cache_helpers.mjs');
    const context = vm.createContext({ queueMicrotask, setTimeout, WeakMap, WeakSet, Set,
        MutationObserver: class { observe() {} disconnect() {} } });
    const source = await readFile(new URL('../web/director_vsr_cache.js', import.meta.url), 'utf8');
    const script = new vm.SourceTextModule(source, { context });
    await script.link(async path => {
        if (path.endsWith('app.js')) return new vm.SyntheticModule(['app'], function () {
            this.setExport('app', { graph: { _nodes: [] }, registerExtension(value) { registered = value; } });
        }, { context });
        return new vm.SyntheticModule(Object.keys(helpers), function () {
            for (const [key, value] of Object.entries(helpers)) this.setExport(key, value);
        }, { context });
    });
    await script.evaluate();
    return registered;
}

test('cache extension does not hook unrelated native nodes', async () => {
    const registered = await extension();
    class NativeSampler { onConnectionsChange() { return 7; } }
    const original = NativeSampler.prototype.onConnectionsChange;
    registered.beforeRegisterNodeDef(NativeSampler, { name: 'KSampler' });
    assert.equal(NativeSampler.prototype.onConnectionsChange, original);
});

test('Director aliases cannot install repeated connection wrappers', async () => {
    const registered = await extension();
    class Director { onConnectionsChange() { return 7; } }
    registered.beforeRegisterNodeDef(Director, { name: 'MiniMaxH3Director' });
    const installed = Director.prototype.onConnectionsChange;
    registered.beforeRegisterNodeDef(Director, { name: 'ComfyMiniMaxH3Director' });
    assert.equal(Director.prototype.onConnectionsChange, installed);
});
