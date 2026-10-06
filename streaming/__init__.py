"""streaming/ 包：流式传送带的家（deep module）。

接口只有一个 async 生成器函数（见 sse.py），背后收着四件事里的三件
（issue 01 范围）：逐块序列化、收尾 [DONE]、上游生成器显式收尾。
块间隔超时（gap timeout）是 issue 03 的第四件事，随做随加——
接口不因此变，测试缝也不因此多。
"""
