"""keys/：虚拟 key 的存储与校验（w4-keys-readme issue 01）。

虚拟 key 的签发、校验与撤销全部藏在 SQLite 单表后面，只经 `KeyStore` 的
create / verify / scope_of / revoke / list 接口读写——明文凭据只在 create 的返回值里出现
一次，库里永远只有 hash（spec：库泄露 ≠ 凭据泄露）。与 billing 账本共用同一个
SQLite 文件、各管各表、`CREATE TABLE IF NOT EXISTS` 自治建表（spec 决定）。
scope 在 store 只存取不解释（scope_of 读原文），白名单判定语义在 `keys/scope.py`
（w4 票 03 兑现"语义归 03 授权层"）。模块分工见各文件 docstring。
"""
