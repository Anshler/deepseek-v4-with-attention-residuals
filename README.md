# DeepSeek-V4-Pro + Attention Residuals

A conceptual modification of [DeepSeek V4 Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) that incorporates [MoonShot's Attention Residuals](https://arxiv.org/abs/2603.15031) mechanism.

**This is not a runnable model — concept only.**

### How it works

1. **Block grouping.** Layers are partitioned into blocks (e.g., `block_size=4` sublayer-units = 2 transformer layers). Within a block, standard residuals accumulate. At block boundaries, the accumulated tensor is promoted to a completed block and the accumulator resets.

2. **Depth mixing.** Each sublayer (ATTN or FFN) has a learnable pseudo-query vector `[d]` and a key-norm. Given a list of source tensors (completed blocks + current partial block), it computes:
   ```
   scores = pseudo_query · RMSNorm(sources) * d^(-0.5)
   weights = softmax(scores)
   input = Σ weights_i * source_i
   ```
   Pseudo-queries are zero-initialized, giving uniform depth weights at startup.

3. **Two independent mixing patterns.** The attention sublayer and FFN sublayer each have their own pseudo-query + key-norm pair, allowing them to learn different depth-routing strategies (e.g., ATTN prefers recent blocks, FFN draws from earlier semantic representations).

4. **State tracking.** The `Transformer` maintains three pieces of state across layers:
   - `blocks: list[Tensor]` — completed block representations
   - `partial_block: Tensor | None` — the currently accumulating block
   - `counter: int` — sublayer count; triggers promotion when `counter >= block_size`

