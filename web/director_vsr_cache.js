import { app } from '../../scripts/app.js';
import { rayDirectorWithVsr, labelVsrUnverified } from './director_vsr_cache_helpers.mjs';

// No cache HTTP interception, media decoding, timers that poll forever, or
// changes to the author's widgets. Only qualify an otherwise unsafe claim.
const states = new WeakMap();
const hooked = new WeakSet();
const pending = new Set();
let scheduled = false;
const relevantTypes = new Set(['MiniMaxH3Director', 'ComfyMiniMaxH3Director',
    'MiniMaxH3DirectorRefine', 'RayMMH3DirectorModelAdapter', 'SolAttnV100', 'TESpeedMiniMaxH3']);
function attach(node) {
    const body = node?._mmxFirstPassCacheUI?.body;
    if (!body || states.get(node)?.body === body) return;
    states.get(node)?.observer.disconnect();
    const state = { body, original: null, rendered: null };
    const update = () => {
        const graph = node.graph ?? app.graph;
        const director = (graph?._nodes ?? []).find(candidate => {
            const input = (candidate.inputs ?? []).find(item => item.name === 'refine');
            const link = graph.links?.get?.(input?.link) ?? graph.links?.[input?.link];
            return String(link?.origin_id ?? link?.[1]) === String(node.id);
        });
        const text = body.textContent ?? '';
        if (rayDirectorWithVsr(graph, director)) {
            const rendered = labelVsrUnverified(text);
            if (rendered !== text) {
                state.original = text;
                state.rendered = rendered;
                body.textContent = rendered;
            }
        } else if (state.original != null && text === state.rendered) {
            body.textContent = state.original;
            state.original = state.rendered = null;
        }
    };
    state.observer = new MutationObserver(update);
    state.observer.observe(body, { childList: true, characterData: true, subtree: true });
    state.update = update;
    states.set(node, state);
    update();
}

function refresh(graph = app.graph) {
    for (const node of graph?._nodes ?? []) {
        attach(node);
        states.get(node)?.update();
    }
}

function schedule(graph) {
    if (!graph) return;
    pending.add(graph);
    if (scheduled) return;
    scheduled = true;
    queueMicrotask(() => {
        scheduled = false;
        for (const item of pending) refresh(item);
        pending.clear();
    });
}

app.registerExtension({
    name: 'Raylight.DirectorVsrCacheStatus',
    afterConfigureGraph() { refresh(); },
    nodeCreated(node) {
        if ((node.comfyClass ?? node.type) === 'MiniMaxH3DirectorRefine') setTimeout(() => attach(node), 0);
    },
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!relevantTypes.has(nodeData?.name) || hooked.has(nodeType.prototype)) return;
        hooked.add(nodeType.prototype);
        const changed = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (...args) {
            const result = changed?.apply(this, args);
            schedule(this.graph);
            return result;
        };
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function (...args) {
            states.get(this)?.observer.disconnect();
            states.delete(this);
            return removed?.apply(this, args);
        };
    },
});
