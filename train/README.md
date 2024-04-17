# Training

## How to train

1. Clone MiniCPM huggingface project to `$PROJ_DIR/pretrained`

```bash
git clone https://huggingface.co/openbmb/MiniCPM-2B-dpo-bf16
```
2. Change a tokenizer setting in `tokenizer_config.json`

```json
"add_eos_token": true
```

3. Create a `checkpoint` folder inside `$PROJ_DIR/train` folder to save a model checkpoint

```bash
mkdir $PROJ_DIR/train/checkpoint
```

4. Run train script

```bash
chmod +x train.sh
./train.sh
```

## Applied Techniques

The following training techniques are applied.

1. Gradient Accumulation : Contrastive models require large batch size for training stability
2. LoRA : rank 8
3. Mixed Precision Training : bf16
4. Learning Rate Scheduler : CosineAnnealingLR
5. Data Parallel Distributed Training

## Batch Size

Having `d` GPU devices, `b` batch size per GPU and `k` gradient accumulation steps is equivalent to training with batch size of `d*b*k`.

## TODO

- Train the best model
- Train from more data sources with more instructions