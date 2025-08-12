### Flux LoRA (Kontext) 训练与 Demo

基于 PyTorch Lightning and difusser，对 `black-forest-labs/FLUX.1-Kontext-dev` 进行 LoRA 微调与推理演示。

### 快速开始
- **安装环境**: 见 `docs/installation.md`
- **准备数据集**: 见 `docs/dataset.md`
- **启动训练**:
```bash
python flux_lora/train_lightning.py \
  --config configs/kyc_flux_training_example.yaml \
  --devices -1 --strategy ddp_find_unused_parameters_true \
  --precision bf16-mixed --accumulate_grad_batches 2
```
- **启动 Web Demo**:
```bash
python flux_lora/tools/web_demo.py \
  --models_dir output_models/kyc_flux \
  --base_model black-forest-labs/FLUX.1-dev \
  --dtype bf16 --port 7860
```

提示: 若使用脚本 `flux_lora/scripts/run_training_and_demo.sh`，无需创建符号链接。

### 文档导航
- `docs/installation.md`: 安装与环境要求
- `docs/dataset.md`: 数据集 CSV 规范与预计算缓存
- `docs/config.md`: 配置文件说明与 CLI 覆盖
- `docs/training.md`: 训练与日志/输出
- `docs/demo.md`: Web Demo 使用
- `docs/faq.md`: 常见问题排查
- `docs/project_structure.md`: 项目结构与代码位置

### 目录结构（简）
- `flux_lora/`: 训练、数据集、Lightning 模块与工具脚本
- `configs/`: YAML 示例配置
- `output_models/`: 训练输出（运行后生成）
- `docs/`: 详细文档

### 许可证
见 `LICENSE`。


