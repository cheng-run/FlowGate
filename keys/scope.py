"""scope 解释层（w4 票 03）：把 scope 原文判成"这个 model 放不放行"。

为什么与 store 分家（store 只存取不解释，票 01 既定纪律）：存取是"列里躺着
什么"，解释是"白名单语义怎么读"——两件事的变更理由不同（加列/改 SQL 不该
碰到判定口径；改口径也不该动存取）。本文件就是 01 注释里预告的"03 授权层"，
纯函数、零 IO、不 import 任何具体适配器（边界检查咬得到的地方）。
判定词汇以 CONTEXT.md 为准：scope=虚拟 key 上的模型白名单，`*` 表示全部；
本层的失败形状叫授权失败（403），与认证失败（401）严格三分（见类 docstring）。
"""


class AuthorizationError(Exception):
    """授权失败的统一信号：403 的唯一失败形状（app/ 异常处理器翻状态码）。

    为什么构造只收 model、不收任意文案：与 AuthenticationError 同一纪律——
    文案在构造里生成，调用方塞不进私货；但**不**钉死统一文案（401 才钉）——
    story 9 明文要求 403 点名被拒的 model（"客户端不必猜"），点名的是调用方
    自己发来的字符串，不泄库内任何信息，与 401 防枚举（统一文案）各有其理：
    401 的答案"你有没有身份"对陌生人必须含糊，403 的答案"你缺哪个 model 的权"
    对已认证调用方必须可操作。
    为什么不放 app/：与 AuthenticationError / RateLimitError 同一纪律——失败
    形状住在语义所属的深模块（scope 判定与拒绝是一体的），HTTP 状态码翻译
    集中在 app/ 的异常处理器。
    """

    def __init__(self, model: str) -> None:
        """只收被拒 model 名——文案唯一出口，点名 model 不回显 key（story 9）。"""
        super().__init__(f"授权失败：模型 {model} 不在该虚拟 key 的 scope 内")


def check_scope(scope: str | None, model: str) -> None:
    """按 scope 白名单判定请求的 model：不在名单抛 AuthorizationError，放行返回 None。

    口径（票面 03 定死并钉进测试）：
    - `*` = 不限 model（CONTEXT.md scope 词条）；列表里混进 `*` 段（如 "m1,*"）
      同样按通配读——通配符的语义就是"不再枚举"；
    - **空 = 不限**，与 `*` 同义：建 key 忘填 scope 是运维手滑，不是"想签一把
      全拒的哑弹"——宽侧缺省对使用者更安全（可用性），真正的收紧动作是填名单；
    - None = 查无（scope_of 对不存在的 key_id 返回 None），**fail-closed 拒绝**：
      与"空=不限"严格分家——身份凭空消失绝不能落进宽侧（story 2 的 fail-closed
      传统：空库=拒绝一切）；
    - 列表按逗号拆段、剥空白、丢空段（" m1 , ,m2 "读作 {m1, m2}）：store 原样
      存取的宽容是本层的活，存取层不解释（票 01 纪律）。
    为什么只判不取数：scope 原文与 model 都是调用方给的纯字符串，判定是纯函数
    ——好测（本文件对应 tests/test_keys_scope.py 直调）、无 IO、无隐藏状态。
    """
    if scope is None:
        # 行级：查无=fail-closed——先于一切宽侧口径，防御位不能被"空=不限"吃掉
        raise AuthorizationError(model)
    # 行级：逗号拆段、剥空白、丢空段——解释层负责读得懂 store 原样存的文本
    entries = {segment.strip() for segment in scope.split(",")}
    entries.discard("")
    if not entries or "*" in entries:
        return  # 行级：空（含纯空白）=不限、`*` 通配——两条宽侧口径在此放行
    if model not in entries:
        raise AuthorizationError(model)  # 行级：名单穷举后未命中——点名拒绝
