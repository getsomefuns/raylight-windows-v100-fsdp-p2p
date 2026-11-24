"""The public graph must use native VSR inputs inside our Director scope."""
import json
from pathlib import Path


def test_director_vsr_example_has_connected_native_pack_and_ray_model():
    path=Path(__file__).resolve().parents[1]/'example_workflows/Minimax_H3_Director_DualV100_RelayVSR.json'
    assert path.is_file(), 'Missing user-loadable Director VSR workflow'
    graph=json.loads(path.read_text(encoding='utf-8'))
    nodes={str(node['id']):node for node in graph['nodes']}
    links={str(link[0]):link for link in graph['links']}
    director=next(node for node in nodes.values() if node['type']=='MiniMaxH3Director')
    def source(node,name):
        item=next(item for item in node['inputs'] if item['name']==name)
        assert item['link'] is not None, f'Unwired {node["type"]}.{name}'
        return nodes[str(links[str(item['link'])][1])]
    adapter=source(director,'model')
    assert adapter['type']=='RayMMH3DirectorModelAdapter'
    vsr=source(director,'vsr')
    assert vsr['type']=='MiniMaxH3DirectorRelayVSR'
    assert source(vsr,'model')['type']=='UNETLoader'
    assert source(vsr,'vae')['type']=='VAELoader'
    # Director already invokes the dual decoder in our scope. No second
    # standalone decoder should be wired after H3 output and double-upscale it.
    assert not any(node['type']=='RayRelayVSRDecode' for node in nodes.values())
    assert 'Minimax_H3_REF2VA_Raylight.jpg' in json.dumps(graph,ensure_ascii=False)


def test_public_example_has_reproducible_turbo4_settings():
    path=Path(__file__).resolve().parents[1]/'example_workflows/Minimax_H3_Director_DualV100_RelayVSR.json'
    graph=json.loads(path.read_text(encoding='utf-8'))
    for node in graph['nodes']:
        if node['type'] in ('MiniMaxH3Director','MiniMaxH3DirectorRelayVSR'):
            assert node['widgets_values_named']['seed']==42
            assert node['widgets_values_named']['control_after_generate']=='fixed'
            assert 'randomize' not in node['widgets_values']
            if node['type']=='MiniMaxH3Director':
                assert node['widgets_values_named']['steps']==4
                assert node['widgets_values_named']['cache_frames_codec']=='ffv1'
                assert node['widgets_values_named']['total_frames']==56
                timeline=json.loads(node['widgets_values_named']['timeline_data'])
                assert timeline['totalFrames']==56
                assert timeline['segments'][0]['frameCount']==56
