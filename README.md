# Language-Model-STS-CFT

[Paper Coming Soon...] [[Hugging Face 🤗](https://huggingface.co/collections/trapoom555/small-lms-text-embedding-663b3ec87527788a577f6852)]

This project aims to improve text embedding of smaller Language Models (LMs) up to 2B parameters using the contrastive fine-tuning technique. Specifically, the InfoNCE loss is utilized as a training objective.

$$\min  - \log \frac{e^{\text{sim}(\textbf{h}_i, \textbf{h}_i^+) / \tau}}{\sum_i \left( e^{\text{sim}(\textbf{h}_i, \textbf{h}_j^+) / \tau }+ e^{\text{sim}(\textbf{h}_i, \textbf{h}_j^-) / \tau} \right)}$$

where $\textbf{h}_i$ denotes an embedding vector of a premise $x_i$, $\tau$ denotes a temperature and $\text{sim}(\textbf{h}_i, \textbf{h}_i^+)$ computes the cosine similarity between embedding vectors $\textbf{h}_i$ and $\textbf{h}_i^+$.

We employ LoRA as our parameter-efficient fine-tuning technique in order to reduce the memory requirement.

## Embedding Extraction

- Every prompt will be appended by the [EOS] token.
- The embedding vector will be extracted from hidden states at the last layer of this [EOS] token.


## Dataset

We utilize the processed NLI dataset (273K) with our synthetic dataset (63K) for finetuning the model 

## Training 
Use train_0.py and set the paths to fine tune your model for embedding. 
We used qwen2-0.5-it 0.5 billion parametor model for fine tuning. 
