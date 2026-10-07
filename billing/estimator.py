"""自建 token 估算器：粗粒度启发式，误差可量化（issue 06 的结算兜底）。

为什么不引 tiktoken：计费在"核心机制一律自建"红线清单内（AGENTS.md 依赖节）——
估算器的误差是**我们**的工程量（live 对照官方 usage 报误差百分比，issue 07 的
数字出处），换成现成库这个考点就没了。为什么接受粗粒度：估算是官方 usage 缺失时
的兜底，不是要跟官方抢精确——口径透明（下述公式）比假装精确更诚实。
"""

import re

# CJK 字符的分类（U+4E00–U+9FFF 汉字区）：一个 CJK 字符≈1 token（比拉丁字符密得多）。
# 字面区间旁必须标码位：编辑器里看不出"一-鿿"是哪到哪，码位就是走读的拐杖
_CJK = re.compile("[一-鿿]")


def estimate_tokens(text: str) -> int:
    """对整段文本估算 token 数：CJK 每字 1，其余每 4 字符 1（向上取整），空文本 0。

    公式（测试手算样例的独立真源）：tokens = cjk 字数 + ceil(非 CJK 字符数 / 4)。
    为什么对**整段**算、不按块加总：SSE 块的切法是任意的，逐块估算再相加会把
    每块的取整误差累加放大（OpenAI 的块默认不带计数，spec 点名"不可按块加总"）——
    收尾对整段文本 tokenize 一次，误差只有一份。
    为什么非 CJK 按 4 字符 1 token：英文常见 token（词/词素）平均 3~4 字符，
    这是最粗也最好讲的一档近似；CJK 每字 1 是 tokenizer 的常见行为（一字一token 上下）。
    """
    cjk = len(_CJK.findall(text))  # 行级：数 CJK 字——每个≈1 token
    other = len(text) - cjk  # 行级：其余字符（拉丁/数字/标点/空白）统一按 4:1 折算
    return cjk + (other + 3) // 4  # 行级：ceil(other/4) 的整数写法——不足 4 字也记 1
