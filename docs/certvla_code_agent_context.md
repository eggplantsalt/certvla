
# CertVLA 代码接力上下文文档（给编程 Agent）

## 0. 这份文档的用途

你是一个负责实现代码的编程 agent。你的任务不是重新发明论文 idea，而是基于这份文档，把一个已经高度收敛的研究方案逐步实现为可运行、可训练、可验证的代码框架。

你必须遵守以下原则：

1. 不要推翻当前主线，不要重新想题。
2. 不要把项目实现成 planner + verifier + memory 的系统拼装。
3. 不要把项目实现成 full symbolic predicate planner / PDDL 系统。
4. 不要把项目实现成“更大的 memory bank”论文。
5. 不要把中间层实现成自然语言 plan。
6. 不要过度工程化。只实现当前研究主线需要的最小系统。
7. 不要一次性做完所有模块。必须按阶段和模块推进。
8. 在任何阶段，如果某个实现决策会改变论文主张，必须先停下来说明，不要擅自改。
9. 默认以现有 OpenVLA / OpenVLA-OFT 风格代码为底座，尽量在其基础上做最小侵入式扩展。
10. 默认先支持最稳的 v1：单图 RGB、chunk-level continuous action、单 state token、半结构化 slots、certificate gap、short-horizon local repair。

## 1. 项目目标

目标不是造一个新的大模型，而是在 OpenVLA / OFT 风格底座上实现一个新的中间状态与闭环训练框架。

项目核心命题：

一个好的 VLA 内部状态，不是“记住更多历史”的状态，也不是“更容易预测动作”的状态；它应该是一个对下一段动作的结构化后果可证实的最小充分状态。

英文核心句：

A good VLA state should be the minimal state sufficient to certify the structured consequence of the next action chunk.

换句话说，这个项目要实现的是：

- 一个递归任务状态 z_t
- 一个当前任务状态读出 s_t
- 一个局部后果证书 c_t
- 一个条件化在 z_t 与 c_t 上的动作 chunk 生成器
- 一个执行后可计算的 certificate gap Gamma_t
- 一个利用 gap 做 tracking / drift detection / local repair 的闭环

## 2. 当前最终主线（不要改）

请把下面这条逻辑视为固定主线：

现有 VLA 在长程任务中脆弱，不只是因为缺少 planner、memory 或 verifier，而是因为其内部状态没有被定义为一个对局部动作后果负责的对象。我们的方法重新定义 VLA 的内部状态，使其同时支持当前状态读出、局部后果预测、动作生成和执行后验证，从而把 tracking、drift detection 和 local repair 统一到同一个状态-证书闭环中。

一句话总结：

我们不是给 OpenVLA 多接一个 planner 或 verifier，而是把 VLA 的内部状态重定义为“对下一段动作后果可证实的最小充分状态”，并让 state、certificate、action、gap 构成同一个闭环。

## 3. 已经明确否定、不要再回头的路线

### 3.1 不做 planner + executor + verifier 系统拼装

不能把系统实现成：
- 上层 planner
- 中层 memory
- 下层 verifier / repair

我们的主角不是系统拼装，而是 state definition。

### 3.2 不把主线写成更强 memory

我们承认长程任务需要历史，但主角不是 memory architecture，而是“什么信息值得进入任务状态”。

### 3.3 不做 full symbolic planning

我们允许半结构化的 slots / certificates，但不做：
- full predicate universe
- PDDL operator planning
- classical symbolic planner

### 3.4 不把 certificate 做成自然语言

certificate 必须是结构化对象，而不是 text plan。

### 3.5 不做多模块自然缝合

不能把实现写成：
- CaSPer 模块
- PoCo 模块
并排拼起来。

必须把它们统一到一个对象中：certifiable task state abstraction。

## 4. 数学对象与符号系统

### 4.1 递归任务状态

z_t = Phi(z_{t-1}, o_t, l)

其中：
- o_t: 当前观测（v1 默认单图 RGB；可选 proprio 以后再扩展）
- l: 语言指令
- z_{t-1}: 上一个 chunk 的持久任务状态

当前 v1 决策：
- 只使用 1 个 persistent state token
- 不做多 state-token memory bank
- 不做外部大 memory module 作为主角

更新形式可参考 gated update：

tilde z_t = f_theta(o_t, l, z_{t-1})
g_t = sigma(W_g [tilde z_t ; z_{t-1}] + b_g)
z_t = g_t * tilde z_t + (1 - g_t) * z_{t-1}

### 4.2 当前任务状态读出

hat s_t = R_phi(z_t)

注意：
- state readout 只能从 z_t 读
- 不允许直接从原始视觉 token 旁路读状态
- 否则 z_t 会退化成装饰

### 4.3 局部后果证书

hat c_t = Q_psi(z_t)

证书定义在 certificate slots 上，使用 per-slot 的 role/value 对象：

c_t = {(u_t^j, g_t^j)} for j in J_cert

其中：
u_t^j in {advance, preserve, ignore}

若 u_t^j = advance，则再预测 chunk 末目标值 g_t^j

最终实现中推荐使用概率形式：
q(u_t^j | z_t)
q(g_t^j | z_t, u_t^j = advance)

### 4.4 动作 chunk

hat A_t = pi(z_t, hat c_t, o_t)

更成熟的 v1 结构推荐“两路动作头”：
- 粗动作分支：由 z_t + c_t 决定动作语义
- 细残差分支：允许访问当前观测，做几何修正

形式可以写成：

A_t^coarse = pi_c(z_t, hat c_t)
Delta A_t^fine = pi_f(o_t, z_t, hat c_t)
hat A_t = A_t^coarse + lambda_res * Delta A_t^fine

### 4.5 执行后证书偏差

执行 chunk 后，重新得到下一时刻状态读出 hat s_{t+H}，计算：

Gamma_t = Gap(hat c_t, hat s_t, hat s_{t+H})

这个量统一承担：
- task progress tracking
- drift detection
- local repair trigger

## 5. slot 设计（v1）

### 5.1 总原则

slot 不是全局 scene graph，而是：

固定词表 + instruction 角色绑定 + validity mask

即：
- 词表固定
- 当前任务中的 target / goal / source / articulation object 由 instruction 绑定
- 不适用的 slot 用 validity mask 屏蔽

### 5.2 slot 家族

J = J_E union J_R union J_C

- J_E: enabling / transit slots
- J_R: result / latch slots
- J_C: confidence / observability slots

### 5.3 v1 slot vocabulary（推荐）

1. ee_target_proximity
2. hand_occupancy
3. target_contact
4. target_goal_proximity
5. support_relation
6. containment_relation
7. articulation_progress
8. orientation_alignment
9. completion_latch
10. task_visible_confidence

### 5.4 slot 值域建议

- Binary:
  - target_contact
  - completion_latch

- Categorical:
  - hand_occupancy in {empty, target, other}
  - support_relation in {none, on_goal_support, on_other_support}
  - containment_relation in {none, in_goal_container, in_other_container}

- Continuous / ordinal in [0,1]:
  - ee_target_proximity
  - target_goal_proximity
  - articulation_progress
  - orientation_alignment

- Confidence:
  - task_visible_confidence in [0,1]

### 5.5 为什么 hand_occupancy 不等于 in_hand_target

因为它必须区分：
- 手空
- 手里拿着目标
- 手里拿着别的东西

“没抓住”和“抓错了东西”是两种不同的偏移，必须可分。

### 5.6 为什么需要 completion_latch

completion_latch 不是简单重复 support / containment 结果，而是表示：

某个关键关系一旦被任务认定为已完成，就进入“应被保持”的里程碑状态。

它的作用是让“不要把已经做对的部分又做坏”进入状态定义，而不是仅靠经验规则。

## 6. instruction 角色绑定

teacher parser 一侧允许显式 role parse：

r(l) = {target, goal, source, articulation, tool}

但在线 policy 一侧不建议再单独预测 role head。

当前默认原则：
- role 只用于离线构造监督标签
- 在线模型只看观测、语言和前一状态
- instruction encoder 隐式吸收 role 语义

## 7. teacher supervision 的来源

### 7.1 状态标签

训练样本按 chunk 组织：

tau_t = (o_t, l, A_{t:t+H-1}, o_{t+H})

teacher parser / simulator 生成：
- s_t
- s_{t+H}
- validity mask m_t^j
- confidence alpha_t^j

推荐：
- 仿真 / LIBERO / 可重放环境：优先使用 oracle 状态导出标签
- 真实机器人数据：使用离线 pseudo-label parser
- parser 只在训练前离线跑，不进入测试时控制环

### 7.2 goal signature

对每条成功 episode，构造末尾 goal signature：

s^star = Aggregate(s_{T-K:T})

注意：
- 这是弱的全局参照
- 不表示 chunk 要直接预测最终状态
- 只是帮助判断局部变化是否有任务效用

## 8. certificate 自动构造（最终版本）

### 8.1 advance 的原则

advance 不能只看“变没变”，也不能只看“离终点近没近”。

必须综合：
- 局部变化 delta
- 持久性 rho
- 终态效用 upsilon
- 对未来结果推进的支持 eta

### 8.2 对结果类 slot j in J_R

定义：
delta_t^j = d_j(s_t^j, s_{t+H}^j)

rho_t^j =
(1/L) * sum_{ell=1..L} 1[d_j(s_{t+H}^j, s_{t+H+ell}^j) < epsilon_j]

upsilon_t^j =
d_j(s_t^j, s^{star j}) - d_j(s_{t+H}^j, s^{star j})

判定：
u_t^j = advance iff
delta_t^j > tau_delta
and
(rho_t^j > tau_rho or upsilon_t^j > tau_upsilon)

### 8.3 对使能类 slot j in J_E

它们不一定直接朝终态单调推进，所以引入未来支持度 eta：

eta_t^j =
max_{ell in [1, L]} 1[
    exists k in J_R :
    d_k(s_{t+H+ell-1}^k, s^{star k}) -
    d_k(s_{t+H+ell}^k, s^{star k}) > tau_R
]

判定：
u_t^j = advance iff
delta_t^j > tau_delta and eta_t^j = 1

### 8.4 advance 的目标值

若 u_t^j = advance，则：
g_t^j = s_{t+H}^j

也就是说，证书的目标是局部 chunk 末状态，不是整任务最终状态。

### 8.5 preserve 的原则（非常重要）

preserve 不能完全靠数据自动挖。

最终定稿为：
- 数据挖 advance
- 结构先验定 preserve

#### latch-preserve
P_t^latch =
{ j in J_R | completion_latch_j = 1 and j not in A_t }

其中 A_t 是当前 chunk 的 advance 集。

#### support-preserve
P_t^support = Pi(s_t, A_t)

Pi 是一小组通用规则，用来描述：
为了成功完成当前 advance 集，哪些 enabling 条件必须保持。

例如：
- 搬运阶段必须保持 hand_occupancy = target
- 从打开的容器中取物时必须保持足够 articulation_progress
- 放置阶段某些接触 / 对齐不能提前破坏

最终：
P_t = P_t^latch union P_t^support

u_t^j = preserve iff j in P_t

### 8.6 ignore

其余 slot 一律：
u_t^j = ignore

## 9. 损失函数（当前定稿版）

### 9.1 状态读出损失

L_state =
sum_j m_t^j * alpha_t^j * ell_j(hat s_t^j, s_t^j)

### 9.2 role 损失

由于 ignore 极多，采用 focal CE：

L_role =
sum_{j in J_cert} m_t^j * alpha_t^j * FocalCE(hat u_t^j, u_t^j)

### 9.3 advance 目标值损失

L_goal =
sum_{j: u_t^j = advance}
m_t^j * alpha_{t+H}^j * ell_j(hat g_t^j, s_{t+H}^j)

### 9.4 动作 chunk 损失

L_act =
(1/H) * sum_{k=0..H-1} ||hat a_{t+k} - a_{t+k}^star||_1

### 9.5 结构一致性损失

advance:
L_adv-cons =
sum_{j: u_t^j = advance} d_j(hat s_{t+H}^j, hat g_t^j)

preserve:
L_pre-cons =
sum_{j: u_t^j = preserve} d_j(hat s_t^j, hat s_{t+H}^j)

总：
L_cons = L_adv-cons + lambda_pre * L_pre-cons

### 9.6 certificate dependence loss（必须保留）

构造错误证书 c_t^-，要求正确证书下动作更接近专家动作：

e^+ = ||A_t^star - pi(z_t, hat c_t, o_t)||_1
e^- = ||A_t^star - pi(z_t, c_t^-, o_t)||_1

L_dep = max(0, m_dep + e^+ - e^-)

目的：
防止动作头无视 certificate，把 certificate 退化成解释头。

### 9.7 consequence-aligned counterfactual loss

#### nuisance-preserving pair
这些变化不该改变 state / certificate：
- 背景
- 光照
- 无关 distractor
- 轻微视角变化

z_t^+ = Phi(z_{t-1}, o_t^+, l)
hat s_t^+ = R(z_t^+)
hat c_t^+ = Q(z_t^+)

L_inv =
||P z_t - P z_t^+||_2^2
+ state loss on (hat s_t^+, s_t)
+ role / goal consistency on c_t

#### consequence-breaking pair
这些变化应改变 state / certificate：
- target identity
- goal receptacle
- support/containment/articulation relations
- instruction-target alignment

z_t^- = Phi(z_{t-1}, o_t^-, l)
hat s_t^- = R(z_t^-)
hat c_t^- = Q(z_t^-)

L_brk =
max(0, mu - ||P z_t - P z_t^-||_2)
+ state loss on negative state labels
+ role / goal losses on negative certificate labels

总：
L_cf = L_inv + L_brk

### 9.8 总损失

L =
lambda_s * L_state
+ lambda_r * L_role
+ lambda_g * L_goal
+ lambda_a * L_act
+ lambda_c * L_cons
+ lambda_d * L_dep
+ lambda_cf * L_cf

## 10. 测试时闭环与 gap

### 10.1 重新编码执行后状态

执行 chunk 后，得到 o_{t+H}，再编码得到 hat s_{t+H}

### 10.2 概率 role 加权 gap

p_t^{adv,j} = q(u_t^j = advance | z_t)
p_t^{pre,j} = q(u_t^j = preserve | z_t)

gamma_t^j =
p_t^{adv,j} * d_j(hat g_t^j, hat s_{t+H}^j)
+ p_t^{pre,j} * d_j(hat s_t^j, hat s_{t+H}^j)

Gamma_t =
[ sum_{j in J_cert} omega_j * kappa_t^j * gamma_t^j ] /
[ sum_{j in J_cert} omega_j * kappa_t^j + epsilon ]

其中：
- omega_j: slot importance weight
- kappa_t^j: confidence / observability weight

### 10.3 gap 的三重作用

Gamma_t 同时承担：
- progress tracking
- drift detection
- local repair trigger

## 11. repair（当前只做短时局部修复）

若 Gamma_t < tau(c_t):
- continue

否则：
- 以当前实际状态 z_{t+H} 重新出发
- 使用更短的 repair horizon H_r < H
- 重新预测局部 repair certificate 与 repair actions

形式上：
hat c_t^{rep} = Q(z_{t+H})
hat A_t^{rep} = pi(z_{t+H}, hat c_t^{rep}, o_{t+H})

不要一上来做：
- 全局 replanner
- 外部 symbolic planner
- 复杂 retrieval memory 主系统

v1 的目标只是证明：
repair 是 certifiable state 自然导出的能力，而不是外挂模块。

## 12. 推荐的实现边界

### 12.1 第一版必须实现的

1. 单图 RGB 输入
2. chunk-level continuous action
3. 单 persistent state token
4. state readout head
5. certificate head
6. advance / preserve / ignore 机制
7. automatic certificate mining
8. certificate dependence loss
9. counterfactual data path
10. certificate gap
11. short-horizon local repair
12. 训练分阶段调度

### 12.2 第一版不要实现的

1. full symbolic planner
2. 多 state-token 大 memory
3. 自然语言 certificate
4. 大型 retrieval memory 作为主角
5. multi-view / proprio / wrist 全家桶
6. full global world model
7. 从零训练 OpenVLA / OpenX 级大模型
8. planner + verifier + memory 系统拼装

## 13. 推荐的仓库目录结构（建议实现，不是绝对强制）

假设在现有 OpenVLA / OpenVLA-OFT 风格代码库中集成，建议新增一层清晰的模块边界，而不是大面积改底座。

建议目录如下：

project_root/
  docs/
    certvla_context.md
    progress.md
    implementation_plan.md
  configs/
    certvla/
      data/
        base.yaml
        libero_oracle.yaml
        real_pseudo.yaml
      model/
        openvla_oft_certvla.yaml
      train/
        stage1_state.yaml
        stage2_certificate.yaml
        stage3_policy.yaml
        stage4_counterfactual.yaml
      inference/
        repair.yaml
  certvla/
    __init__.py
    slots/
      schema.py
      metrics.py
      role_sets.py
      preserve_rules.py
    data/
      chunking.py
      state_labels.py
      certificate_mining.py
      counterfactuals.py
      parser_interfaces.py
    model/
      state_token.py
      state_readout.py
      certificate_head.py
      action_head.py
      certvla_wrapper.py
    training/
      losses.py
      curriculum.py
      sched_sampling.py
      batch_types.py
    inference/
      gap.py
      thresholding.py
      repair.py
      rollout.py
    utils/
      logging.py
      vis.py
      debug_checks.py
  tests/
    test_slot_schema.py
    test_certificate_mining.py
    test_preserve_rules.py
    test_gap.py
    test_model_shapes.py
    test_stage_configs.py

原则：
- 尽量新增独立模块，而不是到处直接魔改底座
- 对底座侵入越小越好
- 优先 wrapper + composable heads 的方式

## 14. 推荐的开发节奏（必须分阶段）

### Phase 0：只做规划，不动代码
- 阅读仓库
- 明确当前 OpenVLA / OFT 代码入口
- 明确数据流、模型流、训练脚本入口
- 输出 implementation_plan.md
- 不要写代码

### Phase 1：只实现数据 / 标签层
- slot schema
- state labels
- certificate mining
- preserve rules
- counterfactual sample builder
- 单元测试

### Phase 2：只实现模型层
- persistent state token
- state readout head
- certificate head
- action head wrapper
- forward shape tests

### Phase 3：只实现训练层
- losses
- curriculum
- scheduled sampling
- stage configs
- smoke test

### Phase 4：只实现 inference 闭环
- certificate gap
- thresholding
- short-horizon local repair
- logging and visualization

### Phase 5：再做集成和清理
- 配置
- 运行脚本
- progress 文档
- 最小 demo

不要一开始尝试“大一统全部实现”。

## 15. 你必须遵守的编码风格和行为规则

1. 先读文件，再判断，不许对没读过的代码胡乱猜。
2. 优先最小侵入式集成，不要大规模重写 OpenVLA 底座。
3. 默认先做最小可行版本，不要提前为未来需求建大量抽象。
4. 不要因为测试需要就硬编码任务或 slot 值。
5. 不要为了“快过测试”牺牲通用性。
6. 如果某个设计会改变论文主张，必须先停下来说明。
7. 实现每个模块后要更新 docs/progress.md，记录：
   - 已完成内容
   - 未完成内容
   - 已知风险
   - 下一步
8. 除非明确要求，否则不要顺手加“好看但不必要”的功能。

## 16. 当前最重要的“成功标准”

v1 成功，不等于所有实验都做完，而是至少满足以下实现层标准：

1. 存在一个明确的递归状态 z_t
2. state readout 与 certificate 只能从 z_t 读
3. certificate 能被自动挖出来
4. preserve 不是纯统计挖掘，而是结构约束
5. 动作头被明确约束为依赖 certificate
6. 能计算 certificate gap
7. gap 能驱动短时局部 repair
8. 所有模块都有最基本的单元测试 / smoke test
9. 代码结构清晰，方便后续实验扩展

## 17. 你当前最应该做的事

如果你刚接手此项目，正确顺序不是开始写代码，而是：

1. 阅读当前代码库
2. 阅读这份上下文文档
3. 输出一份 implementation_plan.md
4. 将整个项目分成 Phase 0–5
5. 明确第一阶段只实现什么，不实现什么
6. 等用户确认后，再做分模块实现

不要直接跳进编码。
