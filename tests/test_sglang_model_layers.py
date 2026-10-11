"""全部上游注册入口的层覆盖；候选源码及预期为 AI 预先手写。"""

import hashlib
import json
import re

import pytest

from examples.sglang_layers import runner
from examples.sglang_layers.builders import PROFILES
from examples.sglang_layers.catalog import (
    COMPONENT_LAYERS,
    CONFIG_LAYERS,
    LAYER_BY_KEY,
    LAYERS,
    MODEL_LAYERS,
    MULTIMODAL_LAYERS,
)
from examples.sglang_layers.upstream import (
    MANIFEST,
    PREFIX,
    REVISION,
    ROOT,
    SOURCES,
    component_registry,
    layer_path,
    model_registry,
    multimodal_registries,
)

# 上游快照仍使用 Flax 的旧 .value 属性；只过滤这一条弃用提示，保留其他告警。
pytestmark = pytest.mark.filterwarnings(
    r"ignore:'\.value' access is now deprecated:DeprecationWarning"
)


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.name)
@pytest.mark.parametrize("spec", LAYERS, ids=lambda s: s.key)
def test_upstream_layer_and_ai_rewrites(spec, profile):
    results = runner.verify(spec, profile)
    assert [r["actual"] for r in results] == ["equivalent", "different"]
    # 除 DSpark 的方法展开外，BF16 正例必须实际改变原始方程列表。
    # Router 权重固定 FP32，输入也为 FP32 时类型准备会消去。
    if profile.name == "bf16" and spec.key not in ("dspark", "qwen3_moe_gate"):
        assert results[0]["raw_order_changed"]
    assert json.loads(json.dumps(results))[1]["witness"]["inputs"]


def test_registry_coverage_is_exact():
    assert set(model_registry()) == set(MODEL_LAYERS)
    assert set(model_registry("multimodal/models/")) == set(MULTIMODAL_LAYERS)
    assert set(component_registry()) == set(COMPONENT_LAYERS)
    names = set().union(*[set(keys) for keys in multimodal_registries().values()])
    assert names == set(CONFIG_LAYERS)
    covered = set().union(
        *(
            set(m.values())
            for m in (MODEL_LAYERS, MULTIMODAL_LAYERS, CONFIG_LAYERS, COMPONENT_LAYERS)
        )
    )
    assert covered == set(LAYER_BY_KEY)
    yaml_names = set()
    for path in SOURCES:
        if path.endswith(".yaml"):
            text = (ROOT / (path + ".txt")).read_text()
            for matched in re.findall(r"^\s*model_class:\s*([^#\n]+)", text, re.MULTILINE):
                yaml_names.update(name.strip() for name in matched.split(","))
    assert yaml_names <= set(COMPONENT_LAYERS)
    assert len(yaml_names) == 14


def test_snapshot_is_complete_and_unmodified():
    tree = json.loads((ROOT / "upstream-model-tree.json").read_text())
    assert tree["revision"] == REVISION
    for entry in tree["files"]:
        data = (ROOT / (entry["path"] + ".txt")).read_bytes()
        # Git blob 对象摘要来自 GitHub tree API，独立于本地生成的 SHA-256 表。
        blob = b"blob " + str(len(data)).encode() + b"\0" + data
        assert hashlib.sha1(blob).hexdigest() == entry["sha"]
    for entry in MANIFEST["files"]:
        data = (ROOT / (entry["path"] + ".txt")).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        assert len(data) == entry["bytes"]


@pytest.mark.parametrize("name", MODEL_LAYERS)
def test_model_really_constructs_selected_layer(name):
    target = LAYER_BY_KEY[MODEL_LAYERS[name]]
    assert layer_path(model_registry()[name], name, target.path, target.class_name), name


@pytest.mark.parametrize("name", MULTIMODAL_LAYERS)
def test_multimodal_entry_really_constructs_selected_layer(name):
    target = LAYER_BY_KEY[MULTIMODAL_LAYERS[name]]
    path = model_registry("multimodal/models/")[name]
    assert layer_path(path, name, target.path, target.class_name), name


@pytest.mark.parametrize("name", COMPONENT_LAYERS)
def test_pipeline_component_really_constructs_selected_layer(name):
    target = LAYER_BY_KEY[COMPONENT_LAYERS[name]]
    path, local_name = component_registry()[name]
    assert path == "multimodal/manager/stage.py" and PREFIX + target.path in SOURCES
    assert layer_path(path, local_name, target.path, target.class_name), name


def test_incorrect_ai_prediction_fails(monkeypatch):
    spec = LAYER_BY_KEY["llama"]
    positive, negative = runner.FAMILIES[spec.family]
    monkeypatch.setitem(runner.FAMILIES, spec.family, (negative, positive))
    with pytest.raises(AssertionError):
        runner.verify(spec, PROFILES[0])


def test_oracle_rejects_matching_but_wrong_base_and_candidate(monkeypatch):
    spec = LAYER_BY_KEY["llama"]
    positive, negative = runner.FAMILIES[spec.family]
    monkeypatch.setattr(runner, "baseline", lambda spec, m, *x: negative(m, *x))
    monkeypatch.setitem(runner.FAMILIES, spec.family, (negative, positive))
    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        runner.verify(spec, PROFILES[0])
