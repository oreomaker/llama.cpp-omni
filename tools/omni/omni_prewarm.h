#pragma once

// init 预热：代表输入 graph 预热 + vocoder 预热。
// 纯 init 模块：只在 omni_init 里、duplex_prepare 之后调用一次，
// 不参与 duplex 调度（不碰 frame/worker/队列）。无条件跑，正常启动不需要 env。
struct omni_context;

void omni_prewarm(struct omni_context * ctx_omni);
