# -*- coding: utf-8 -*-
"""把后端包一层，让 `move_base` 变成「写意图」而不是「写底盘」。

=== 为什么用代理而不是改 ManipulateSkill ===
ManipulateSkill 的 VLA 路径里已经有一句
`backend.move_base(chunk[10], chunk[11], chunk[12])` —— 那是**直写**。
要让它走仲裁器有两条路：
  A. 改技能，注入 mux，把直写换成 set_vla；
  B. 包一层后端，move_base 的语义整体变成「VLA 意图」。

选 B。理由不是省事，而是**覆盖面**：除了 ManipulateSkill，任何未来
写 `move_base` 的地方（调试脚本、DAgger 回放、别的技能）都会自动被
仲裁，不会再出现「某个新技能绕过仲裁直接抢底盘」的漏洞。技能层代码
一行不改，切换部署形态仍然只是换一个构造。

注意：仲裁器本身拿的是**原始**后端（它才是真正的写者），代理只交给
L2 技能层。
"""


class MuxBackend(object):
    """RobotBackend 的鸭子类型代理：只改底盘语义，其余全部透传。"""

    def __init__(self, backend, mux):
        self._backend = backend
        self._mux = mux

    # ---------- 被改写的三个 ----------
    @property
    def supports_base(self):
        return bool(getattr(self._mux, "supports_base", False))

    def move_base(self, vx=0.0, vy=0.0, wz=0.0):
        """不再是「立即写底盘」，而是「登记 VLA 意图」。

        所有权不在 VLA 时，这次写入只进缓冲区，不产生任何底盘动作 ——
        这正是交班窗口需要的行为：模型还在推理，但车必须听导航的。
        """
        self._mux.set_vla(vx, vy, wz)
        return {"ok": True, "err": None}

    def stop_base(self):
        self._mux.set_vla(0.0, 0.0, 0.0)
        return {"ok": True, "err": None}

    # ---------- 其余全部透传 ----------
    def __getattr__(self, item):
        # 只会在正常属性查找失败时被调用，所以上面三个不会被绕进来
        return getattr(self._backend, item)

    @property
    def name(self):
        return "Mux(%s)" % getattr(self._backend, "name", "?")

    def __repr__(self):
        return "<MuxBackend %s owner=%s>" % (self.name,
                                             getattr(self._mux, "owner", "?"))
