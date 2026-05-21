# Mini-BLIP2 图像描述生成复现要求

## 1. 复现论文

本次复现参考论文：

**BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models**

论文地址：

https://arxiv.org/abs/2301.12597

PDF：

https://arxiv.org/pdf/2301.12597

---

## 2. 复现目标

实现一个简化版 BLIP-2，用于完成 **Image Captioning 图像描述生成**。

任务形式：

```text
输入：一张图片
输出：一句英文图片描述 caption
```

本次不要求完整复现 BLIP-2 的大规模预训练，只要求完成一次轻量化学习复现：能够读取数据、搭建模型结构、完成训练流程。

---

## 3. 数据集要求

使用数据集：**Flickr8k Image Captioning Dataset**

Kaggle 地址：

https://www.kaggle.com/datasets/adityajn105/flickr8k

本次复现只要求使用 Flickr8k 的前 **200 张图片** 及其对应 caption。

---

## 4. 模型结构要求

需要按照 BLIP-2 的核心思想实现如下结构：

```text
Image
  ↓
Frozen Vision Encoder
  ↓
Trainable Mini Q-Former
  ↓
Projection Layer
  ↓
Frozen Language Decoder
  ↓
Caption
```

训练时主要训练：

```text
Mini Q-Former + Projection Layer
```

冻结：

```text
Vision Encoder + Language Decoder
```

---

## 5. 推荐模型组件

### 视觉编码器

推荐使用：

```text
openai/clip-vit-base-patch32
```

地址：

https://huggingface.co/openai/clip-vit-base-patch32

### 语言解码器

推荐使用：

```text
facebook/opt-125m
```

地址：

https://huggingface.co/facebook/opt-125m

---

## 6. 训练要求

基本要求：

```text
任务：image captioning
损失函数：cross entropy loss
训练模块：Mini Q-Former + Projection Layer
冻结模块：Vision Encoder + Language Decoder
数据量：Flickr8k 前 200 张图片
```

---

## 7. 最低完成标准

只要完成以下内容，即认为达到最低复现要求：

1. 能够读取 Flickr8k 前 200 张图片及对应 caption；
2. 能够搭建 Mini-BLIP2 模型结构；
3. 能够完成训练流程。

---

## 8. 最终提交内容

最终提交内容包括：

1. **实现代码**  
   包含数据读取、模型定义、训练脚本、生成/测试脚本。代码需要能从 Flickr8k 前 200 张图片开始训练。

2. **简短实验报告**  
   说明复现论文、使用数据集、模型结构、冻结了哪些模块、训练了哪些模块、训练设置，以及最终效果。

3. **训练过程记录**  
   提供训练 loss 变化或关键日志截图，说明模型确实完成了训练流程。无需达到原论文指标。

4. **生成结果展示**  
   至少展示 3—5 张测试图片，并给出真实 caption 与模型生成 caption，用于说明模型能够根据图片生成文本。

---

## 9. 过程记录与防作弊要求

为了确认作业是本人独立完成、而不是一次性让 AI 把整份代码生成出来，本次复现**必须**满足以下两条过程性要求。两条都不满足将视为未完成。

### 9.1 AI 对话全过程记录

要求使用 **entir.io**（或同类可分享的对话录制工具）记录与 AI（ChatGPT / Claude / Gemini 等）的**全部**开发对话，并在提交时附上可访问链接。

要求：

```text
- 覆盖范围：从读数据、搭模型到训练、调 bug、写报告的全过程
- 不能只录最后一段"成品对话"，中间的试错、报错、追问都要保留
- 链接需可访问（公开或对老师/助教开放）
```

示例（提交时这样写即可）：

```text
AI 对话记录：https://entir.io/s/xxxxxx
使用模型：Claude / ChatGPT
对话时长：累计约 3 小时，分 5 次会话
```

### 9.2 Git 小步提交

要求**每完成一个小模块就提交一次 commit**，禁止"一次性把整个项目 push 上去"。

合格的 commit 粒度示例：

```text
feat: 加载 Flickr8k 前 200 张图片与 caption
feat: 接入 CLIP ViT-B/32 作为 frozen vision encoder
feat: 实现 Mini Q-Former 模块（含 learnable queries）
feat: 添加 projection layer 对齐到 OPT 词向量空间
feat: 接入 frozen OPT-125m 作为语言解码器
feat: 实现训练 loop 与 cross entropy loss
fix: 修复 caption tokenizer padding 与 attention mask 不一致
feat: 添加 caption 生成脚本（greedy / beam search）
docs: 补充实验报告与 loss 曲线
```



提交时一并附上：

```text
git log --oneline 的截图或文本输出
仓库地址（GitHub / Gitee 均可）
```
