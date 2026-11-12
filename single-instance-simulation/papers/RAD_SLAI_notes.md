# RAD & SLAI 阅读笔记

## 论文关注点
RAD 与 SLAI 分别从系统执行节奏和请求/资源分配角度优化 LLM serving。阅读时需要区分“何时启动一次 Decode iteration”与“这一轮选择哪些请求/如何分配资源”，不能把二者混成一个 Priority 算法。

## RAD：iteration/tiling 侧
RAD 关注计算执行组织与 tiling/iteration 周期参数，目标是在执行效率和服务时限之间取得平衡。理解论文时重点核对理论最优条件、周期参数的定义，以及其假设适用范围。

## SLAI：动态调度侧
SLAI 关注调度决策，包括请求优先级、动态 offset/resource allocation 等机制。其价值在于根据运行状态调整请求执行安排，而不是仅使用固定的静态队列顺序。

## 本项目映射
- 当前固定 chunk 策略只作为基线，不等同 RAD。
- 当前 `priority_deadline` 是简化优先级排序，不是 SLAI 原算法。
- 后续可以分别设计 iteration timing/tiling controller 与 request selection/resource allocation controller，再做组合消融。

## 关键实验
1. 只改变 iteration 周期/tiling，不改变请求排序。
2. 只改变请求优先级/offset，不改变 chunk。
3. 两者联合。
4. 对比 TTFT、TPOT/TBT、吞吐、Goodput 与调度开销。

## 复核提醒
论文公式、定理充分条件和参数符号应以论文原文为准。本笔记不替代逐式证明；在正式报告中引用定理时应保留原论文页码与符号定义。
