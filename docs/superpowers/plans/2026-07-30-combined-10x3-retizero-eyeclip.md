# Combined 10×3 RetiZero 与 EyeCLIP 补齐 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Combined 实验矩阵改为无 RETFound、EyeCLIP 与 RetiZero 完整覆盖的 10×3=30，并让 RetiZero Phase B/C 可以原生训练和恢复。

**Architecture:** 在 `RetiZeroAdapter` 内集中管理 CIR 可训练参数与新旧 checkpoint 兼容；`clip_fine_tune.py` 只负责模型分派和训练策略调用；`combiner_train.py` 通过适配器统一入口加载 Phase C 权重。命令矩阵测试按阶段和模型身份验证，不加载大型真实权重。

**Tech Stack:** Python 3.9、PyTorch、pytest、Bash、OpenAI CLIP、Transformers。

## Global Constraints

- 只修改 Combined 命令区，UWF/IDRiD 命令逐字节不变。
- `run_validate_lora.sh` 保持不变，仅参考其旧 checkpoint 约定。
- Combined 每种配置只运行 1 次，全部使用 GPU 0 与 `nohup`。
- RETFound 暂时从 Combined 移除，不删除其历史代码和历史命令。
- 使用测试驱动开发：每个生产行为先写失败测试并确认失败。

---

### Task 1: 锁定 10×3 命令矩阵

**Files:**
- Modify: `tests/test_combined_commands.py`
- Modify: `命令.sh`

**Interfaces:**
- Consumes: `# COMBINED_COMMANDS_BEGIN` / `# COMBINED_COMMANDS_END` 标记区域。
- Produces: `_phase_commands()` 测试辅助函数；A/B/C 各 10 条的 Combined 命令。

- [ ] **Step 1: 写失败测试**

在 `tests/test_combined_commands.py` 中按 `# Phase A/B/C:` 边界拆分命令，断言：

```python
assert {phase: len(lines) for phase, lines in phase_commands.items()} == {
    "A": 10,
    "B": 10,
    "C": 10,
}
assert not any("RETFound" in line for line in commands)
for model_tag in ("eyeclip", "retizero"):
    assert all(
        sum(model_tag in line.lower() for line in phase_commands[phase]) == 1
        for phase in ("A", "B", "C")
    )
```

同时把总数、Combiner 数、fine-tune 数和唯一日志数更新为 30、20、10、30。

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest tests/test_combined_commands.py -q
```

Expected: FAIL，现有矩阵为 A=8、B=10、C=11，且仍含 RETFound。

- [ ] **Step 3: 最小化修改 Combined 命令**

在 `命令.sh` 的 Combined 标记区：

- Phase A 新增 EyeCLIP frozen + Combiner；
- Phase A 新增 RetiZero base + Combiner，使用 `--retizero-base-path /data0/qrchen/projects/CLIP4Cir/pretrained_models/RetiZero.pth`；
- Phase B 删除 RETFound，新增 RetiZero CIR fine-tune，使用 `--clip-model-name RetiZero --retizero-base-path /data0/qrchen/projects/CLIP4Cir/pretrained_models/RetiZero.pth --encoder both`；
- Phase C 删除 RETFound；
- Phase C RetiZero checkpoint 后缀统一为 `.pt`；
- 更新阶段条数和 10×3 注释。

- [ ] **Step 4: 运行命令测试确认通过**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest tests/test_combined_commands.py -q
```

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add tests/test_combined_commands.py 命令.sh
git commit -m "test: define Combined 10x3 command matrix"
```

### Task 2: RetiZero 适配器训练策略与 checkpoint 兼容

**Files:**
- Create: `tests/test_retizero_adapter.py`
- Modify: `src/retizero_adapter.py`

**Interfaces:**
- Produces: `RetiZeroAdapter.configure_cir_finetuning() -> list[str]`
- Produces: `RetiZeroAdapter.load_checkpoint(path) -> tuple[int, float]`
- Preserves: `RetiZeroAdapter.load_lora_checkpoint(path) -> tuple[int, float]`
- Produces: `RetiZeroAdapter.visual` property exposing the actual vision module.

- [ ] **Step 1: 写训练参数策略失败测试**

用不加载真实大模型的 tiny RetiZero fixture 构造 adapter，断言调用 `configure_cir_finetuning()` 后：

```python
assert trainable_names
assert all(
    any(token in name for token in (
        "linear_a_q", "linear_b_q", "linear_a_v", "linear_b_v",
        "projection_head_vision", "projection_head_text",
    ))
    for name in trainable_names
)
assert not any("text_model.model" in name for name in trainable_names)
assert not any("lora_vit" in name and "linear_" not in name for name in trainable_names)
```

具体允许项为 vision LoRA 的 `linear_a_*` / `linear_b_*`、`projection_head_vision`、`projection_head_text`；基础 ViT、BioClinicalBERT 与无效温度参数保持冻结。

- [ ] **Step 2: 运行训练策略测试确认失败**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest tests/test_retizero_adapter.py::test_configure_cir_finetuning_only_enables_lora_and_projection_heads -q
```

Expected: FAIL，方法尚不存在。

- [ ] **Step 3: 实现真实 visual 与训练策略**

在 `src/retizero_adapter.py`：

```python
@property
def visual(self):
    return self.retizero.vision_model

def configure_cir_finetuning(self):
    self.requires_grad_(False)
    for layer in [*self.retizero.vision_model.model.w_As,
                  *self.retizero.vision_model.model.w_Bs]:
        layer.requires_grad_(True)
    self.retizero.vision_model.projection_head_vision.requires_grad_(True)
    self.retizero.text_model.projection_head_text.requires_grad_(True)
    return [name for name, param in self.named_parameters() if param.requires_grad]
```

并在初始化时给实际 vision module 设置 `input_resolution=224`、`output_dim=512`。

- [ ] **Step 4: 运行训练策略测试确认通过**

Run 同 Step 2。Expected: PASS。

- [ ] **Step 5: 写两类 checkpoint 加载失败测试**

新增两个测试，分别用 tiny adapter 的现有 `state_dict()` 和 vision encoder 的 `state_dict()` 构造 checkpoint：

```python
def test_load_checkpoint_restores_full_clip4cir_adapter(tmp_path):
    adapter = make_tiny_adapter()
    expected = {key: value.clone() for key, value in adapter.state_dict().items()}
    checkpoint = tmp_path / "cir.pt"
    torch.save({"epoch": 7, "RetiZeroAdapter": expected}, checkpoint)
    for parameter in adapter.parameters():
        parameter.data.zero_()
    assert adapter.load_checkpoint(checkpoint) == (7, -1)
    assert all(torch.equal(adapter.state_dict()[key], value) for key, value in expected.items())


def test_load_checkpoint_accepts_legacy_classification_lora(tmp_path):
    adapter = make_tiny_adapter()
    expected = {
        key: torch.full_like(value, 2)
        for key, value in adapter.retizero.vision_model.model.state_dict().items()
    }
    checkpoint = tmp_path / "legacy.pth"
    torch.save({
        "epoch": 4,
        "mean_ACC": 0.8,
        "state_dict": {f"img_encoder.{key}": value for key, value in expected.items()},
    }, checkpoint)
    assert adapter.load_checkpoint(checkpoint) == (4, 0.8)
    assert all(
        torch.equal(adapter.retizero.vision_model.model.state_dict()[key], value)
        for key, value in expected.items()
    )
```

第一类保存 `{"epoch": 7, "RetiZeroAdapter": adapter.state_dict()}`；第二类保存 `{"epoch": 4, "mean_ACC": 0.8, "state_dict": {"img_encoder.<key>": value}}`。分别断言完整恢复和旧 key 映射。

- [ ] **Step 6: 运行 checkpoint 测试确认失败**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest tests/test_retizero_adapter.py -q
```

Expected: FAIL，统一加载入口尚不存在。

- [ ] **Step 7: 实现统一 checkpoint 加载**

实现 `load_checkpoint()`：

- `RetiZeroAdapter` key：清理可选 `module.` / `clip_model.` 前缀并严格加载完整 adapter；
- `state_dict` 且存在 `img_encoder.*`：复用旧 vision-only 映射；
- 其他格式抛出包含实际顶层 key 的 `ValueError`；
- `load_lora_checkpoint()` 保留为调用 `load_checkpoint()` 的兼容别名。

- [ ] **Step 8: 运行 adapter 测试确认通过**

Run 同 Step 6。Expected: PASS。

- [ ] **Step 9: 提交**

```bash
git add tests/test_retizero_adapter.py src/retizero_adapter.py
git commit -m "feat: support CIR tuning checkpoints for RetiZero"
```

### Task 3: 将 RetiZero 接入 Phase B/C 入口

**Files:**
- Create: `tests/test_retizero_training_integration.py`
- Modify: `src/clip_fine_tune.py`
- Modify: `src/combiner_train.py`

**Interfaces:**
- Produces: `_is_retizero_model_name(model_name: str) -> bool`
- Produces: `_uses_raw_text_inputs(model_name: str) -> bool`
- Consumes: `RetiZeroAdapter.configure_cir_finetuning()`
- Consumes: `RetiZeroAdapter.load_checkpoint(path)`

- [ ] **Step 1: 写模型分派和文本输入失败测试**

通过 monkeypatch 替换 `RetiZeroAdapter` 为 tiny fake，断言：

```python
model, preprocess = _load_model_for_finetune(
    "RetiZero", {"retizero_base_path": "/tmp/base.pth", "force_rgb": True}
)
assert isinstance(model, FakeRetiZeroAdapter)
assert _uses_raw_text_inputs("RetiZero")
```

测试还要断言 `configure_cir_finetuning()` 被训练策略调用，且非 `both` 模式给出明确错误。

- [ ] **Step 2: 运行集成测试确认失败**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest tests/test_retizero_training_integration.py -q
```

Expected: FAIL，RetiZero 当前会落入 `clip.load`，且不会使用原始字符串。

- [ ] **Step 3: 实现 fine-tune 分派**

在 `src/clip_fine_tune.py`：

- 增加 `_is_retizero_model_name()` 与 `_uses_raw_text_inputs()`；
- `_load_model_for_finetune()` 用 `--retizero-base-path` 构造 `RetiZeroAdapter`；
- `_maybe_load_custom_clip_weights()` 跳过 RetiZero；
- RetiZero 只接受 `--encoder both`，并调用 `configure_cir_finetuning()`；
- FIQ/CIRR 文本分支都将 RetiZero 作为原始字符串列表处理；
- CLI 和 hyperparameter 字典增加 `--retizero-base-path`。

- [ ] **Step 4: 实现 combiner checkpoint 恢复**

在 `src/combiner_train.py` 的 RetiZero 分支：

- 始终用 `retizero_base_path` 构造 base adapter；
- 如果有 `clip_model_path`，调用 `clip_model.load_checkpoint(clip_model_path)`；
- 删除重复 probe 整个大 checkpoint 的旧判断逻辑。

- [ ] **Step 5: 运行集成测试确认通过**

Run 同 Step 2。Expected: PASS。

- [ ] **Step 6: 提交**

```bash
git add tests/test_retizero_training_integration.py src/clip_fine_tune.py src/combiner_train.py
git commit -m "feat: integrate RetiZero into CIR fine-tuning"
```

### Task 4: 完整验证与本地合并

**Files:**
- Verify: `src/retizero_adapter.py`
- Verify: `src/clip_fine_tune.py`
- Verify: `src/combiner_train.py`
- Verify: `命令.sh`
- Verify: `tests/`

**Interfaces:**
- Consumes: Tasks 1–3 的完整结果。
- Produces: 可合并且验证通过的 feature branch。

- [ ] **Step 1: Python 语法检查**

```bash
/data0/qrchen/miniconda3/envs/clip4cir/bin/python -m py_compile \
  src/retizero_adapter.py src/clip_fine_tune.py src/combiner_train.py
```

- [ ] **Step 2: Shell 语法检查**

```bash
bash -n 命令.sh
bash -n run_validate_lora.sh
```

- [ ] **Step 3: 完整测试**

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest -q
```

Expected: 0 failures。

- [ ] **Step 4: 检查需求与差异**

```bash
git diff main...HEAD --check
git diff --stat main...HEAD
git status --short
```

确认 Combined 为 30 条、UWF/IDRiD hash 测试通过、未修改两个用户未跟踪文件。

- [ ] **Step 5: 按已确认工作流本地合并回 main**

在主工作区执行：

```bash
git merge --ff-only feat/combined-10x3-retizero-eyeclip
```

合并后在 `main` 再运行完整测试，并报告提交、测试数量及命令矩阵。
