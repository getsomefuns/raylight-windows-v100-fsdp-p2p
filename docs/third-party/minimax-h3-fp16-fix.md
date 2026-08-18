# MiniMax H3 FP16 fix attribution

Raylight's opt-in MiniMax H3 safe-FP16 adapter is based on the numerical
strategy published by
[`Amduraznak/minimax-h3-fp16-fix`](https://github.com/Amduraznak/minimax-h3-fp16-fix),
pinned locally at commit `b09897c`.

The adapted concepts are:

- FP32 condition projection and residual accumulation;
- FP16 attention and MLP branches;
- exact power-of-two rescaling around attention `out_proj` (`64`) and MLP
  `fc2` (`256`).

Raylight's implementation differs from the upstream drop-in module by making
the behavior an explicit `fp16_h3_safe` loader mode, installing it inside every
Ray worker before model/FSDP construction, validating the ComfyUI MiniMax API,
and preserving the Windows CUDA P2P plus FP8-FSDP storage path. It does not
modify ComfyUI's global supported dtype list.

## Upstream license

MIT License

Copyright (c) 2026 Amduraznak

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
