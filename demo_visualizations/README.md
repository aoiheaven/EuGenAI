# Demo Visualizations - 演示可视化图表

**生成时间**: 2024年11月9日  
**目的**: 展示医学多模态思维链模型的可解释性评估

---

## 📁 文件列表

| 文件名 | 说明 | 大小 |
|--------|------|------|
| `1_attention_heatmap.png` | 注意力热力图 - 展示模型关注区域 | 2.3 MB |
| `2_chain_of_thought.png` | 思维链可视化 - 展示推理过程 | 8.1 MB |
| `3_reliability_diagram.png` | 可靠性图 - 验证置信度校准 | 284 KB |
| `4_attention_localization.png` | 注意力定位对比 - 与专家对比 | 3.1 MB |
| `5_deletion_insertion.png` | 删除/插入曲线 - 验证重要性 | 315 KB |
| `6_evaluation_dashboard.png` | 综合评估仪表板 - 全面总览 | 659 KB |
| `可视化图表解释说明.md` | **详细解释文档** | 18 KB |

---

## 🎯 快速导航

### 👀 想看模型关注什么？
→ 查看 `1_attention_heatmap.png`

### 🧠 想了解推理过程？
→ 查看 `2_chain_of_thought.png`

### 🎲 想验证可信度？
→ 查看 `3_reliability_diagram.png`

### 👨‍⚕️ 想对比专家意见？
→ 查看 `4_attention_localization.png`

### 🔬 想证明科学性？
→ 查看 `5_deletion_insertion.png`

### 📊 想要全面评估？
→ 查看 `6_evaluation_dashboard.png`

---

## 📖 如何使用

### 查看图片
直接双击PNG文件即可在系统默认图片查看器中打开。

### 理解图表
阅读 `可视化图表解释说明.md`，包含：
- 每张图的详细解释
- 图表元素含义
- 如何解读指标
- 科学意义说明

### 重新生成
如果需要修改或重新生成：
```bash
cd /Users/harryw/MyDev/jmm/quiz/explanity
source .venv/bin/activate
python scripts/generate_demo_visualizations.py
```

---

## 🎨 图片预览

### 1. 注意力热力图
![示意](1_attention_heatmap.png)
**展示内容**: 原图 | 热力图 | 叠加视图

### 2. 思维链推理
![示意](2_chain_of_thought.png)
**展示内容**: 5步推理过程，每步都有边界框和注意力分数

### 3. 可靠性图
![示意](3_reliability_diagram.png)
**展示内容**: 置信度vs准确率 + 置信度分布

### 4. 注意力定位
![示意](4_attention_localization.png)
**展示内容**: 专家标注 vs AI注意力 + 对比分析

### 5. 删除/插入曲线
![示意](5_deletion_insertion.png)
**展示内容**: 验证注意力区域的重要性

### 6. 评估仪表板
![示意](6_evaluation_dashboard.png)
**展示内容**: 综合指标 + 多个子图

---

## 📊 关键指标一览

从这些可视化中可以得到的关键指标：

### 分类性能
- ✅ **准确率**: 89.2%
- ✅ **F1分数**: 0.91
- ✅ **AUC-ROC**: 0.94

### 置信度校准
- ✅ **ECE**: 0.032（优秀）
- ✅ **校准质量**: Good

### 注意力质量
- ✅ **重叠度**: 0.87
- ✅ **定位准确率**: 92%
- ✅ **删除AUC**: 0.23（低=好）

### 推理质量
- ✅ **一致性**: 0.78
- ✅ **连贯性**: 0.85
- ✅ **专家一致**: 81%

---

## 💡 使用场景

### 学术论文
- 用于展示模型的可解释性
- 证明医学AI的可信度
- 支持方法学描述

### 临床演示
- 向医生展示AI如何工作
- 建立对AI的信任
- 辅助临床决策

### 监管审批
- 提供安全性证据
- 展示全面的评估
- 满足可解释性要求

### 教学培训
- 医学生学习材料
- AI教育资源
- 案例分析示例

---

## 🔧 技术信息

**生成工具**: Python 3.10 + matplotlib + seaborn + opencv  
**图片格式**: PNG (300 DPI，适合打印)  
**数据**: 合成演示数据（非真实患者数据）  
**许可**: 遵循项目LICENSE

---

## 📞 问题反馈

如有问题或建议，请参考：
- 主README: `../README_zh.md`
- 详细说明: `可视化图表解释说明.md`
- 生成脚本: `../scripts/generate_demo_visualizations.py`

---

**注意**: 这些是演示用的合成数据和可视化。实际使用时，请用真实的医学数据和训练好的模型生成可视化结果。

