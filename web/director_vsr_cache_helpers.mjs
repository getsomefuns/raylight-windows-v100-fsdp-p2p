function source(graph, node, name) {
    const input = (node?.inputs ?? []).find(item => item.name === name);
    if (input?.link == null) return null;
    const link = graph.links?.get?.(input.link) ?? graph.links?.[input.link];
    return graph.getNodeById?.(link?.origin_id ?? link?.[1]);
}

export function rayDirectorWithVsr(graph, director) {
    if (!(director?.inputs ?? []).some(input => input.name === 'vsr' && input.link != null)) return false;
    const pending = [source(graph, director, 'model')];
    const visited = new Set();
    while (pending.length && visited.size < 128) {
        const node = pending.pop();
        if (!node || visited.has(node)) continue;
        visited.add(node);
        if ((node.comfyClass ?? node.type) === 'RayMMH3DirectorModelAdapter') return true;
        for (const input of node.inputs ?? []) {
            if (input.type === 'MODEL' || input.type === '*') pending.push(source(graph, node, input.name));
        }
    }
    return false;
}

export function labelVsrUnverified(text) {
    return text.replace(/^成片匹配：.*$/m,
        '成片匹配：未验证（原生缓存检查未传入 RelayVSR 配置；不影响一采匹配，运行时会核对成片缓存）');
}
