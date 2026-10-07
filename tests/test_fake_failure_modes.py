"""fake 可控失败形态 + 命名区分测试（fallback-ratelimit-billing issue 01，纯测试设施 prefactor）。

为什么先做这个 prefactor：W3 的 fallback/计费测试要能断言"谁挂了、谁接管了、谁被取消了"，
fake 得先会按需失败、按名字区分——"先让改动变容易，再做容易的改动"。
两条缝（spec 预先约定）：Provider 缝（chat/chat_stream 的外部行为：抛什么、吐什么、卡不卡）
+ fake 记录面（ticket 显式指定的观测面：调用/收尾账本，供事后查账断言）。
全部离线：fake 零外网，卡住用例只设等待上限、不真实 sleep。
"""

import asyncio

import pytest

from app.schemas import ChatMessage, ChatRequest
from providers.base import UpstreamError
from providers.fake import FakeProvider


def _make_request() -> ChatRequest:
    """最小合法请求——与 providers/streaming 测试同款，跨文件读起来是同一套语言。"""
    return ChatRequest(
        model="fake-model",
        messages=[ChatMessage(role="user", content="你好")],
    )


# ===== 切片 A：恒失败（fail）——两腿同一失败形状（checklist 1/5）=====


async def test_chat_raises_upstream_error_when_failure_fail() -> None:
    """证明：注入恒失败后非流式腿抛 UpstreamError 族失败——与真实适配器同一失败形状。

    怎么证明：FakeProvider(failure="fail") 调 chat()，断言抛出的是 providers.base.UpstreamError
    （协议词汇）而不是 fake 私有异常。W3 的路由只认这个异常族当换路信号——fake 若抛
    别的类型，fallback 测试测的就不是真实的失败形状，测了也白测。
    """
    with pytest.raises(UpstreamError):
        await FakeProvider(failure="fail").chat(_make_request())


async def test_chat_stream_raises_upstream_error_when_failure_fail() -> None:
    """证明：恒失败的流式腿同样抛 UpstreamError——失败形态注入对两腿都生效（checklist 1）。

    怎么证明：消费 chat_stream 的首块（anext），断言拿到的是 UpstreamError。
    流式腿的失败发生在首块前——正是 W3"可报错窗口/重试窗口"依赖的时机。
    """
    stream = FakeProvider(failure="fail").chat_stream(_make_request())

    with pytest.raises(UpstreamError):
        await anext(stream)


async def test_fail_message_identical_across_streaming_and_nonstreaming() -> None:
    """证明：恒失败的两条腿报同一条错——"非流式与流式两腿失败行为一致"的字面证据（checklist 5）。

    怎么证明：两个独立 fake 实例分别在两腿上失败，断言两条错误消息逐字相等。
    反例是消息里带腿名（"stream 失败"/"chat 失败"）——那客户端两条腿就得各写一套
    分支，与"错误处理永远不需要上游/腿特定分支"（spec 故事 24）相悖。
    """
    with pytest.raises(UpstreamError) as nonstream:
        await FakeProvider(failure="fail").chat(_make_request())

    stream = FakeProvider(failure="fail").chat_stream(_make_request())
    with pytest.raises(UpstreamError) as streaming:
        await anext(stream)

    assert str(nonstream.value) == str(streaming.value)


# ===== 切片 B：空流（empty）——一言不发（checklist 2）=====


async def test_chat_stream_yields_no_chunks_when_failure_empty() -> None:
    """证明：注入空流后流式腿一块不吐——"一言不发"的流式形态（checklist 2）。

    怎么证明：消费 chat_stream 到底，断言收集到的块数为 0（不是"有块但内容空"）。
    这是 issue 03 空流桩的可复用形态：网关在首块前收到"没有下一块"，翻 502——
    fake 提供同款死法，W3 的全链失败用例不必再造桩。
    """
    # 复杂语句（async 推导式）行上：把流跑到底、把块全收集——数出来的块数就是"说了几句"
    chunks = [c async for c in FakeProvider(failure="empty").chat_stream(_make_request())]

    assert chunks == []


async def test_chat_returns_empty_content_when_failure_empty() -> None:
    """证明：空流形态在非流式腿同样一言不发——答复内容为空串（checklist 5：两腿一致）。

    怎么证明：chat() 正常返回（不抛错），但 message.content 为空——流式腿"零块收场"
    与非流式腿"空答复收场"是同一种死法在两条腿的各自讲法（流可以"不说话地结束"，
    整答必须给个答复对象，就给空内容）。反例是照常回显——那注入就没生效。
    """
    response = await FakeProvider(failure="empty").chat(_make_request())

    assert response.choices[0].message.content == ""


# ===== 切片 C：卡住（hang）——悬在产出之前，等被取消（checklist 2）=====


async def test_chat_blocks_before_answering_when_failure_hang() -> None:
    """证明：注入卡住后非流式腿悬着不返回——W3"尝试预算内放弃"的仿真对象（checklist 2）。

    怎么证明：并发起 chat() 任务，只等一个上限（0.1s 是兜底不是节奏，零真实 sleep），
    到点断言任务没完成；随后取消它——卡住必须可被取消，否则 W3 的"放弃尝试时显式
    取消"根本落不了地。全程不触网、不睡真实时间。
    """
    # 行级：create_task 用在"等上限观察卡不卡"——不并发起任务，就没法在"还在悬着"
    # 这一刻只等一个上限（同步调用只能等到返回，观察不到卡住本身）
    task = asyncio.create_task(FakeProvider(failure="hang").chat(_make_request()))

    done, _ = await asyncio.wait({task}, timeout=0.1)
    assert not done  # 上限内没返回 = 真的卡住了

    task.cancel()  # 行级：卡住必须可取消——收摊 + 验证"显式取消"能落进 fake
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_chat_stream_blocks_before_first_chunk_when_failure_hang() -> None:
    """证明：注入卡住后流式腿悬在首块之前——与非流式腿同一种死法（checklist 2/5）。

    怎么证明：取首块的任务只等一个上限，断言没取到（卡在首块前，不是半路）；
    然后取消取块任务收摊。流式卡住要卡在**首块前**才符合 W3 的"可报错窗口"故事——
    卡在半路是另一码事（那要靠闸门逐块放行后停住，既有用例已覆盖）。
    """
    stream = FakeProvider(failure="hang").chat_stream(_make_request())

    # 行级：create_task 把"取首块"放到任务里悬着——才能只等一个上限、断言它没取到
    first_try = asyncio.create_task(anext(stream))
    done, _ = await asyncio.wait({first_try}, timeout=0.1)
    assert not done  # 首块迟迟不来 = 卡在产出之前

    first_try.cancel()  # 行级：取消传播进生成器的挂起点——W3"显式取消"的同款现场
    with pytest.raises(asyncio.CancelledError):
        await first_try


async def test_cancelled_hang_stream_records_cancelled_ending() -> None:
    """证明：卡住的流被显式取消后收尾账本记"被取消"——"谁被取消了"可断言（checklist 4）。

    怎么证明：起一条卡住的流，取消它的取块任务（等价于 W3"放弃尝试时显式取消上游"：
    取消异常穿进生成器的挂起点），断言 stream_endings 恰一条且字面量为 "cancelled"。
    失败注入的流记的是 "failed"（切片 A 已钉）——两种死法账本可区分，
    W3 数"被取消了几个"不会把失败数进去。
    """
    fake = FakeProvider(failure="hang")
    stream = fake.chat_stream(_make_request())

    # 行级：create_task 同上——先让流挂上，再取消它（等上限确认真挂住，不靠猜时机）
    first_try = asyncio.create_task(anext(stream))
    await asyncio.wait({first_try}, timeout=0.1)  # 行级：等它真的挂上（上限兜底，零 sleep）
    first_try.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_try

    assert fake.stream_endings == ["cancelled"]


# ===== 切片 D：命名区分 + 按实例记账（checklist 3/4）=====


async def test_response_carries_instance_name_when_named() -> None:
    """证明：命名实例的答复里带自己的名字——回答面可见"谁服务的"（checklist 3，两腿）。

    怎么证明：fake-a / fake-b 各在非流式腿答一题，断言各自 id 以自己的名字开头；
    再看流式腿：首块的 id 同样带名字（两腿命名面一致，checklist 5 的"行为一致"）；
    最后钉一条默认实例 id 仍以 "fake" 开头——纯加法红线：不命名的用法一字不改。
    """
    response_a = await FakeProvider(name="fake-a").chat(_make_request())
    response_b = await FakeProvider(name="fake-b").chat(_make_request())
    default = await FakeProvider().chat(_make_request())
    stream_first = await anext(FakeProvider(name="fake-b").chat_stream(_make_request()))

    assert response_a.id.startswith("fake-a")
    assert response_b.id.startswith("fake-b")
    assert stream_first.id.startswith("fake-b")  # 行级：流式帧同样带实例名
    assert default.id.startswith("fake")  # 行级：默认名的历史形状不漂移


async def test_calls_recorded_per_instance_when_named() -> None:
    """证明：调用按实例记账、记次数——"谁被调用了几次"可断言（checklist 3/4）。

    怎么证明：fake-a 非流式答一题、fake-b 非流式答一题 + 流式跑一条，断言各自的
    调用账本只含自己的记录（"谁服务的"看谁的账本动了）且次数对得上。
    两实例的账本分开，W3 才能断言"只调了 a 一次、b 零次"这类 fallback 现场。
    """
    fake_a = FakeProvider(name="fake-a")
    fake_b = FakeProvider(name="fake-b")

    await fake_a.chat(_make_request())
    await fake_b.chat(_make_request())
    # 复杂语句（推导式）行上：流式腿在首块被消费时才开工记账（async 生成器语义），
    # 这里跑到底把一条流完整走完
    _ = [c async for c in fake_b.chat_stream(_make_request())]

    assert fake_a.calls == ["chat"]
    assert fake_b.calls == ["chat", "chat_stream"]


async def test_failed_stream_records_failed_ending_per_instance() -> None:
    """证明：恒失败的流收尾记 "failed"、调用记账照记——失败 ≠ 被取消（checklist 4）。

    怎么证明：命名 + 恒失败的实例跑一条流（首块消费即抛），断言调用账本记了一笔、
    收尾账本恰一条且字面量为 "failed"。与卡住用例的 "cancelled" 合看：两种死法
    账本可区分，W3 数"谁被取消了"不会把失败的数进去。
    """
    fake_a = FakeProvider(name="fake-a", failure="fail")
    stream = fake_a.chat_stream(_make_request())

    with pytest.raises(UpstreamError):
        await anext(stream)

    assert fake_a.calls == ["chat_stream"]
    assert fake_a.stream_endings == ["failed"]


def test_unknown_failure_mode_rejected_loudly() -> None:
    """证明：拼错的失败形态响亮报错——注入面配错不静默退化成"正常 fake"。

    怎么证明：传一个不存在的形态值，断言 ValueError 且消息里点名肇事值。
    反例是静默忽略：测试本想让 a 挂掉，fake 却答得好好的，fallback 用例假绿——
    配错要喊响（与装配处"未知 FLOWGATE_PROVIDER 响亮报错"同一纪律）。
    """
    with pytest.raises(ValueError, match="always-fail"):
        FakeProvider(failure="always-fail")


# ===== 切片 E：非流式收尾账（fallback-ratelimit-billing issue 02）=====


async def test_chat_endings_record_completed_failed_and_cancelled() -> None:
    """证明：非流式腿收尾账与流式同一词汇（completed / failed / cancelled）——
    "放弃的尝试显式取消上游（fake 记录'被取消'）"的可断言面（issue 02 checklist）。

    怎么证明：三个实例分别走三条收场路——正常答完、恒失败抛错、卡住被显式取消——
    断言 chat_endings 各记恰好一条且字面量对号入座。为什么非流式也要收尾账
    （issue 01 曾定"非流式无收尾账"）：失败/完成当场以异常/返回值可见，唯有
    "被取消"看不见——fallback 链放弃卡住的尝试时显式 cancel 上游，fake 记下
    "被取消"就是"取消真落进了上游代码、而不是干等它烧钱"的证据。
    """
    ok = FakeProvider()
    await ok.chat(_make_request())

    bad = FakeProvider(failure="fail")
    with pytest.raises(UpstreamError):
        await bad.chat(_make_request())

    # 行级：create_task 让 chat() 悬在挂起点上，才能取消它（同步调用只能等到返回）
    hung = FakeProvider(failure="hang")
    task = asyncio.create_task(hung.chat(_make_request()))
    await asyncio.wait({task}, timeout=0.1)  # 行级：等它真的挂上（上限兜底，零真实 sleep）
    task.cancel()  # 行级：显式取消——fallback 链"放弃尝试"的同款现场
    with pytest.raises(asyncio.CancelledError):
        await task

    assert ok.chat_endings == ["completed"]
    assert bad.chat_endings == ["failed"]
    assert hung.chat_endings == ["cancelled"]
