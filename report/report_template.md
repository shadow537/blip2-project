# Mini-BLIP2 图像描述生成复现实验报告

## 1. 论文信息

- 论文名称：BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
- 论文地址：https://arxiv.org/abs/2301.12597

## 2. 任务说明

本实验复现的任务是图像描述生成 Image Captioning。

输入：图片  
输出：英文 caption

## 3. 数据集

- 数据集名称：Flickr8k
- 数据集地址：https://www.kaggle.com/datasets/adityajn105/flickr8k
- 实际使用数据量：前 200 张图片

## 4. 模型结构

请说明自己的 Mini-BLIP2 结构，例如：

```text
Image → Frozen Vision Encoder → Mini Q-Former → Projection Layer → Frozen Language Decoder → Caption
```

### 4.1 Vision Encoder

填写使用的视觉编码器，例如：`openai/clip-vit-base-patch32`。

### 4.2 Mini Q-Former

说明自己实现的 Mini Q-Former：

- query token 数量：
- hidden size：
- Transformer 层数：
- 是否使用 cross-attention：

### 4.3 Language Decoder

填写使用的语言解码器，例如：`facebook/opt-125m`。

## 5. 训练设置

请填写：

- 训练数据量：
- epoch：
- batch size：
- learning rate：
- optimizer：
- loss function：
- 冻结的模块：
- 训练的模块：

## 6. 训练过程

粘贴训练日志或 loss 变化截图。

示例：

| Epoch | Train Loss |
|---|---:|
| 1 |  |
| 2 |  |
| 3 |  |

## 7. 生成结果展示

至少展示 3—5 个例子。

| 图片编号 | 真实 Caption | 模型生成 Caption |
|---|---|---|
| 1 |  |  |
| 2 |  |  |
| 3 |  |  |

如果方便，可以把图片也插入报告中。

## 8. 总结

请简要说明：

- 是否成功跑通训练；
- 生成效果如何；
- 遇到了什么问题；
- 如果继续改进，可以怎么做。

## 9. AI 对话过程记录

请填写本次复现过程中与 AI 工具的对话记录（对应 requirements.md 第 9.1 节）。

- 录制工具：例如 entir.io
- 对话链接：
- 使用的 AI 模型：例如 Claude / ChatGPT / Gemini
- 累计对话时长 / 会话数：

简要说明 AI 在哪些环节给了帮助、哪些地方是自己独立完成或推翻了 AI 的建议（2—4 句话即可）：

```text
（在这里写）
```

## 10. Git 提交记录

请填写本次复现的代码仓库与提交历史（对应 requirements.md 第 9.2 节）。

- 仓库地址：
- 总 commit 数：

粘贴 `git log --oneline` 输出（或截图）：

```text
（在这里粘贴 git log --oneline）
```
