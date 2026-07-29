<p align="center">
  <img src="../image/vestigia_logo.svg" width="500" alt="Vestigia 标志">
</p>

# Vestigia

[English README](../README.md)

Vestigia 是一个用于构建和比较大语言模型（LLM）**行为指纹**的 Python 工具包。它对模型重复发送固定探针，从每次回答中提取稳定特征，并比较得到的经验分布。

## 项目目的

模型指纹不是模型名称，也不是单条回答，而是在受控实验下观测到的输出分布。实验条件包括：

- 固定的 prompt 与 system instruction；
- 固定的生成参数和 endpoint 配置；
- 探针特定的解析器与特征字段；
- 多次独立采集的模型回答。

借此可以衡量一组未知来源的模型输出，与历史参考模型指纹库之间的行为相似程度。

## 应用场景

- **模型溯源研究**：判断外部采集的输出分布更接近哪个参考模型。
- **网关与部署验证**：检查路由后的 endpoint 行为是否与预期参考部署一致。
- **模型回归监控**：跟踪模型、网关或推理服务栈升级后行为分布的变化。
- **LLM 评测研究**：使用受控探针比较模型倾向，而非只依据单次回答。

Vestigia 通过 LiteLLM 支持 OpenAI-compatible 与 Anthropic 风格的 endpoint，内置“最喜欢的数字”“项目成功率评分”等固定探针，并提供采集、稳定性分析、指纹持久化和离线分布预测能力。

## 文档

- [Usage guide (English)](usage_en.md)
- [使用说明（中文）](usage_cn.md)
