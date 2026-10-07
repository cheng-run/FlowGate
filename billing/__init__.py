"""billing/：两段式计费（fallback-ratelimit-billing issue 06，非流式腿）。

账本两本（用户账 + 内部损耗账）藏在 SQLite 后面，只经 `BillingLedger` 的
"结算 + 查询"接口读写——幂等与不重复扣费从此只有一个可观测面（spec 决定）。
结算口径：官方 usage 优先，缺失则自建估算器对整段文本 tokenize（issue 07 的
流式回填沿用同一口径）。模块分工见各文件 docstring。
"""
