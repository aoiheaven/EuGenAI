# 👁️ EuGenAI - 眼科生成式智能诊断系统

## Eye Ultra-intelligent Generative AI

### *生成式智能眼科诊疗*

[![License](https://img.shields.io/badge/License-Custom-red.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

[English Documentation](README.md)

---

## 项目概述

**EuGenAI** (Eye Ultra-intelligent Generative AI) 是一个专注于**眼科疾病诊断**的生成式AI系统。基于PyTorch深度学习框架，结合视觉Transformer和BERT文本编码器，通过生成式思维链推理提供透明、可验证的诊断过程。

### 为什么选择EuGenAI？

与传统"黑盒"AI不同，EuGenAI提供：
- 👁️ **眼科专精**：针对视网膜病变、青光眼、黄斑疾病优化
- 🤖 **生成式推理**：自动生成类人的诊断推理链
- 🔍 **多病灶分析**：同时检测多个病变（微动脉瘤、渗出、玻璃疣等）
- 🖼️ **多模态融合**：整合眼底照片、OCT、血管造影、临床病史
- 📊 **精确分割**：像素级病灶定位，支持手术规划和疗效监测
- ⚡ **最少标注**：自监督+强化学习减少90%标注需求

### 核心特性

- **多模态学习**：整合医学图像（CT、MRI、X光等）与临床文本（病史、检验结果、体格检查）
- **思维链推理**：明确的逐步推理过程，模拟临床决策思维
- **注意力可视化**：生成注意力热力图，展示模型关注的图像区域和文本片段
- **可解释AI**：提供透明的诊断推理，适合临床验证
- **灵活架构**：支持多种视觉和语言模型backbone

### 架构设计

```
输入:
├── 医学图像 (CT/MRI/X光/超声)
├── 临床文本 (病史、体格检查、实验室检查)
└── 思维链步骤 (基于区域的推理)

模型:
├── 视觉Transformer (图像编码器)
├── BERT (文本编码器)
├── 跨模态注意力 (图像-文本融合)
└── 思维链解码器 (推理生成器)

输出:
├── 诊断预测
├── 置信度分数
├── 注意力热力图
└── 推理解释
```

## 安装

### 前置要求

- Python 3.9 或更高版本
- CUDA 11.8+ (GPU支持)
- [uv](https://github.com/astral-sh/uv) 包管理器

### 使用uv安装

```bash
# 克隆仓库
git clone https://github.com/aoiheaven/EuGenAI.git
cd EuGenAI

# 如果还没有安装uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境并安装依赖
uv venv
source .venv/bin/activate  # Windows系统: .venv\Scripts\activate

# 安装项目（依赖会自动安装）
uv pip install -e .
```

### 传统pip安装

```bash
pip install -e .
```

## 数据格式

框架需要JSON格式的数据。完整的数据结构请参考 `data_format_example.json`。

### 数据结构示例

```json
{
  "sample_id": "MED_001",
  "image": {
    "path": "images/MED_001.jpg",
    "modality": "CT",
    "body_part": "chest"
  },
  "medical_record": {
    "history": "患者病史...",
    "physical_exam": "体格检查所见...",
    "lab_results": "实验室检查结果..."
  },
  "chain_of_thought": {
    "reasoning_steps": [
      {
        "step": 1,
        "action": "观察整体影像",
        "observation": "发现...",
        "region_of_interest": {
          "bbox": [x1, y1, x2, y2],
          "description": "区域描述"
        }
      }
    ]
  },
  "final_diagnosis": {
    "primary": "诊断名称",
    "confidence": 0.92
  }
}
```

## 使用方法

### 训练

```bash
# 使用默认配置训练
python src/train.py --config configs/default_config.yaml

# 从检查点恢复训练
python src/train.py --config configs/default_config.yaml --resume checkpoints/checkpoint_epoch_10.pth
```

### 推理

```bash
# 对单张图像进行推理
python src/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --image path/to/medical_image.jpg \
    --text "患者临床信息..." \
    --output outputs/
```

### 自定义配置

编辑 `configs/default_config.yaml` 可以自定义：

- 模型架构（视觉/文本编码器、隐藏维度）
- 训练超参数（学习率、批量大小、训练轮数）
- 数据增强设置
- 日志和检查点选项

## 项目结构

```
medical-multimodal-cot/
├── configs/
│   └── default_config.yaml       # 训练配置
├── data/
│   ├── train.json                # 训练数据
│   ├── val.json                  # 验证数据
│   └── images/                   # 医学图像
├── src/
│   ├── __init__.py
│   ├── dataset.py                # 数据集和数据加载
│   ├── model.py                  # 模型架构
│   ├── train.py                  # 训练脚本
│   └── inference.py              # 推理和可视化
├── checkpoints/                  # 保存的模型检查点
├── logs/                         # 训练日志
├── outputs/                      # 推理输出
├── data_format_example.json      # 数据格式规范
├── pyproject.toml                # 项目依赖（uv）
├── LICENSE                       # 许可证文件
├── README.md                     # 英文文档
└── README_zh.md                  # 本文件
```

## 模型组件

### 1. 图像编码器
- 基于视觉Transformer (ViT)
- 提取patch级别和全局图像特征
- 支持来自timm库的预训练模型

### 2. 文本编码器
- BERT架构
- 编码临床文本和推理步骤
- 生成上下文词嵌入

### 3. 跨模态注意力
- 图像和文本之间的双向注意力
- 实现视觉和文本信息的细粒度对齐
- 生成可解释的注意力权重

### 4. 思维链解码器
- 顺序处理推理步骤
- 整合区域特定信息
- 产生可解释的诊断推理

## 可视化

框架提供多种可视化工具：

### 注意力热力图
- 在医学图像上叠加注意力权重
- 显示模型关注的区域
- 支持单步和聚合可视化

### 思维链步骤
- 显示逐步推理过程
- 用边界框高亮感兴趣区域
- 将视觉证据与文本观察配对

### 示例代码

```python
from src.inference import MedicalCoTInference

# 初始化推理引擎
engine = MedicalCoTInference('checkpoints/best_model.pth')

# 生成综合报告
engine.generate_report(
    image_path='data/images/case_001.jpg',
    clinical_text='患者病史和检查所见...',
    reasoning_steps=[...],
    output_dir='outputs/case_001/'
)
```

## 评估指标

框架支持以下评估指标：

- **诊断准确率**：诊断标签的分类准确度
- **置信度校准**：预测置信度与实际准确度的一致性
- **注意力定位**：注意力图与真实病灶的重叠度
- **推理有效性**：临床专家对思维链步骤的评价

## 引用

如果您在研究中使用本框架，请引用：

```bibtex
@software{eugenai_2024,
  title={EuGenAI: Eye Ultra-intelligent Generative AI for Chain-of-Thought Ophthalmic Diagnosis},
  author={aoiheaven},
  year={2024},
  url={https://github.com/aoiheaven/EuGenAI}
}
```

## 许可证

**重要提示**：本项目采用自定义限制性许可证。

**未经原作者事先书面同意，禁止：**
- 高校或研究机构使用本代码进行学术发表
- 商业用途
- 用于发表的衍生作品

完整条款请参见 [LICENSE](LICENSE) 文件。

## 贡献

欢迎贡献！请遵循以下步骤：

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启Pull Request

## 致谢

本项目基于：

- [PyTorch](https://pytorch.org/) - 深度学习框架
- [Transformers](https://huggingface.co/transformers/) - 预训练模型
- [timm](https://github.com/huggingface/pytorch-image-models) - 视觉模型
- 医学AI研究社区

## 联系方式

如有问题、issue或合作咨询：

- GitHub Issues: [项目Issues](https://github.com/aoiheaven/EuGenAI/issues)
- 邮箱: 通过GitHub Issues联系

## 开发路线图

- [ ] 支持3D医学图像（CT/MRI体积数据）
- [ ] 多GPU分布式训练
- [ ] 预训练模型权重
- [ ] 基于Web的交互式演示
- [ ] 与DICOM浏览器集成
- [ ] 临床文本的多语言支持

---

**免责声明**：这是一个研究工具，在没有适当验证和监管批准的情况下，不应用于临床决策。

