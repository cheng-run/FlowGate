"""ratelimit/：令牌桶限流的家（W3 issue 05）。

接口是 RateLimiter.acquire(key)——桶里有令牌才放行，空桶抛 RateLimitError；
门卫（FastAPI 依赖）与 429 翻译在 app/，本包不认 HTTP（deep module：
换限流算法只动这一个包，门卫与路由一字不改）。
"""
