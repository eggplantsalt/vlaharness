# Controller-Handoff 自主研究审查

> 审查日期：2026-08-09。本文记录实现后的研究判断，不是实验结果。
> “实现事实”来自当前 checkout 的静态检查；“文献事实”来自所列原始论文；
> “研究假设”和“待验证”均不得写成 established finding。Windows 阶段按任务约束
> 未运行 pytest、导入冒烟、模拟器、GPU、Gate-0 或模型推理。

## A. 当前设计的双视角批判

### 作者视角

Phase A 的主要价值不是某一个分类器，而是把一个可证伪问题接到真实执行边界：
冻结 VLA、部署时可得的连续具身状态、逐步解析控制、每步重新观测、真实正负 VLA
结果、显式成本与经验不确定性、无 privileged online input，并同时保留 Gate-0、
controller-only、Original Harness 和完整 agent 层对照。

当前参考策略在每次观测后比较两类配置成本：立即 handoff；或向若干
target-relative 候选执行至多一个有界解析动作后重新评估。若未来候选更优，只执行
一步并用真实新观测重算，而不是在预测状态上开环走完。这个结构直接检验局部物理
governor 是否比 planner 反复微调更有效。

但方法表述必须收窄。当前实现没有 staging transition model、observation model、
长期 value-to-go 或可信的未来视觉生成。因此它不是已经求解的 optimal stopping、
POMDP 或多步最优控制。准确名称是 **outcome-calibrated myopic
receding-horizon switching（结果校准的单步滚动切换）**。bootstrap 保守分数是经验
不确定性，不是形式化安全保证。

### 最苛刻 reviewer 视角

1. **核心机制已有强近邻。** 成功概率/可达域、技能前置条件、起始状态优化、
   setup/transition policy 和用后续任务成败学习 switch 都有直接先例。不能把
   “learn competence and switch” 本身宣称为新意。
2. **“最优”只是给定标量化成本下的局部 argmin。** failure、VLA、staging、
   hysteresis 权重可能改变结论，必须固定主协议并做 sensitivity。
3. **future-state 近似有限。** 将来 EEF 几何是构造值；未来遮挡、感知、接触和
   reachability 不是。所有 approximation/unavailable 标记都要进入结果解释。
4. **在线选择会改变数据分布。** Gate-0 的受控干预缓解选择偏差，但 governor
   访问的状态分布、candidate lattice 与 Gate-0 仍可能不同。必须做独立
   task/reset/candidate group 的校准与 coverage 诊断。
5. **标签层级不可互换。** primitive success、skill evaluator、官方 task
   termination、truncation、planner finish、RPC/VLA failure、staging failure、labeler
   failure 必须分别保留。
6. **系统严谨性不等于算法贡献。** 更少 LLM calls、可恢复日志、artifact identity
   和 provenance firewall 很重要，但不能替代机制新颖性。
7. **尚无运行或经验依据。** 静态实现只说明实验路径已被编码；任何效果、校准、
   稳定性或速度结论都仍是未知。

### 核心判断

**Problem-Reality：UNCERTAIN。** Harness VLA 提供了真实的 staging/restaging 动机，
但“pre-handoff 连续状态是否产生稳定、可迁移、可校准的 frozen-VLA outcome
gradient”必须先由 Gate-0 证明。

**Novelty：UNCERTAIN（高 residual risk）。** 当前可辩护空间只来自非常具体的系统
组合和实证差异，而不是通用 competence/switching 叙事。最危险的 reviewer 解释是：
“把已有 success-based switching / skill precondition / start-state optimization
换成 VLA，再加工程系统。”

## B. Phase B 候选机制审查

### 1. Conformal risk-controlled handoff

- Problem-Reality：**PASS（形式保证缺口真实）**。现有 bootstrap/quantile 分数没有
  finite-sample 风险保证。
- Novelty：**FAIL**。[Conformal Decision Theory](https://arxiv.org/abs/2310.05921)
  已将“信任 nominal policy 或切换到 safe backup”作为直接应用；
  [Conformal Policy Learning](https://arxiv.org/abs/2311.01457) 更直接地在多个 base
  policy 间用 conformal quantile 切换并给出保证。
- 决定：不新增为 Phase B 主机制。若真实 calibration/risk-coverage 失败，可作为
  有明确既有来源的 baseline/extension，不能重新包装为核心新意。

### 2. Information-gathering / active-perception staging

- Problem-Reality：**UNCERTAIN**。target uncertainty/occlusion 可能使“move to see”
  有价值，但当前服务器 probe 和 Gate-0 尚未证明它是主要失败源。
- Novelty：**FAIL / scope 风险高**。主动改变视角获取 manipulation-relevant 信息是
  成熟方向；例如 [Active Perception and Representation for Robotic Manipulation](https://arxiv.org/abs/2003.06734)
  已用视角变化服务定位和操作，[Real-World Reinforcement Learning of Active
  Perception Behaviors](https://arxiv.org/abs/2512.01188) 在真实操作中学习信息收集。
- 决定：不引入额外 observation dynamics、world model 或 active-perception policy。

### 3. Causal / off-policy counterfactual handoff value

- Problem-Reality：**UNCERTAIN**。只看到已选择 handoff 的结果会产生选择偏差，
  但 Gate-0 已通过受控 candidate intervention 收集正负结果；当前没有证据表明更复杂
  识别是首要瓶颈。
- Novelty：**UNCERTAIN，当前收益不足**。通用 OPE、causal policy learning 和
  doubly robust estimation 已成熟；没有 overlap/ignorability 证据时换 estimator
  也不会自动识别未执行 counterfactual。
- 决定：先使用受控数据、严格 group split、held-out calibration 和 coverage。

### 4. Learned setup/transition policy 或 RoA planner

- Problem-Reality：**UNCERTAIN**。解析 servo 是否覆盖不足要等真实 staging
  failure/碰撞证据，不能因更复杂方法存在就假定问题成立。
- Novelty：**FAIL**。[Learning Setup Policies](https://arxiv.org/abs/2101.09391)
  已学习连接预训练 controller 的 setup policy；[Training Transition Policies via
  Distribution Matching](https://arxiv.org/abs/2110.04357) 用 stay/switch 与后续任务
  成败学习 transition；[Hybrid Systems Neural Control with Region-of-Attraction
  Planner](https://proceedings.mlr.press/v211/meng23a.html) 学习跨 mode RoA 并规划。
- 决定：不把“简单解析 staging + 冻结 VLA”扩大成新控制器学习项目。

### 5. Multi-step world model / POMDP optimal stopping

- Problem-Reality：**PASS（表述缺口）**，effect size **UNCERTAIN**。它能补上当前
  没有 value-to-go 的形式缺口，但尚无证据说明单步重观察不够。
- Novelty：**UNCERTAIN**，scope drift：**FAIL**。完整 world model/POMDP 会改变论文
  问题、数据要求和实现边界。
- 决定：诚实重命名当前方法，不为了数学叙述引入 speculative model。

### 6. Phase B 结论

以上候选没有同时通过 Problem-Reality 和 Novelty 两道门。因此本轮**没有新增
speculative research policy/config**，也没有用更复杂模块制造产出。Phase A 参考方法
保留为值得实验的可证伪假设，而不是 novelty PASS。

## C. 吸收的工程与科学完整性改进

以下改动提高证据质量，但不作为 research-method novelty：

- planner-visible `pi0_pick`/`pi0_doubled` schema 不变；opt-in 时仅透明替换内部
  handler，disabled 时保持 Original Harness handler；
- 正常 CLI 不变；研究 child 另外生成原子、排他的 run-local post-reset identity
  与 completion sidecar，保留 planner exception；
- full-agent summary 严格绑定 transcript、states、sidecars、planner identity、source
  revision、controller identity 和详细 outcome；游离/重复/矛盾记录 fail closed；
- stable outcome key 只绑定 invocation identity，使同一调用的矛盾重试必然碰撞；
- Gate-0 `candidate_id` 与 repeat 无关，`sample_id` 绑定一次执行；resume 不截断
  torn tail，也不接受不一致的 trial/sample/reset/controller；
- 模型 artifact ID 绑定完整 manifest 和 estimator checksum；positive-reference ID
  绑定完整引用、build settings、排除项和 source record IDs；
- scientific trial identity 绑定解析后的 runtime/planner/condition、source revision、
  task/handoff config checksum、checkpoint ID 和 artifact bytes；
- server probe 支持显式 `--require-observed` readiness gate，并核对 operator 提供的
  Pi0.5/SAM3 content-derived checkpoint ID；
- VLA 已执行但 labeler 失败时记录独立 failure mode，并从训练集中明确排除；
- post-hoc oracle 只在 exact matched Gate-0 context 的唯一候选间比较 realized cost，
  明确标记 policy-ineligible；
- 聚合拒绝 execution layer / record scope 混合，task 表保留 condition/method/config，
  ablation plot 拒绝未控制因素混合；
- positive-only reference 只从 materialized train split 构建；三套独立 Gate-0 候选
  cohort 为 candidate-group split 提供可分组件。

这些是当前静态实现事实；是否能在 Linux CUDA/MuJoCo 环境按预期运行仍未验证。

## D. Prior-art / novelty audit

### 保留的窄化参考假设

**机制本质。** agent 先语义承诺一个 contact-rich skill；本地 governor 再用当前
部署观测和真实正负 VLA outcome 模型，反复比较“现在交给 frozen VLA”与“付出一个
有界解析 staging 成本后重评估”。每次 staging 后重新观察，禁止在线 privileged
state。

**问题已有依据但尚未被证明。** [Harness VLA](https://arxiv.org/abs/2607.08448)
本身把 frozen VLA 暴露为可重试 primitive，并由 agent 进行解析 staging/restaging；
这说明系统存在调用条件与 operating-range 问题，但不证明连续局部 governor 优于
planner heuristic。

**最强近邻和实质重叠：**

- [Learning When to Switch](https://arxiv.org/abs/2011.00440) 为目标 controller
  学习成功切换概率并选择 switch region，已覆盖“continuous state + learned
  success estimator + threshold switching”；
- [Hierarchical Policies for Cluttered-Scene Grasping with Latent Plans](https://arxiv.org/abs/2107.01518)
  用 option classifier 判断切到预训练 grasp policy 是否会成功；
- [Training Transition Policies via Distribution Matching](https://arxiv.org/abs/2110.04357)
  直接使用后继 pretrained policy 的真实任务成败训练 stay/switch；
- [Where To Start?](https://proceedings.mlr.press/v205/vosylius23a.html) 学习技能从
  某配置开始能否成功/无碰撞，并优化起始配置后再运动规划过去，与 competence
  projection/start-state optimization 非常接近；
- [Relational Learning for Skill Preconditions](https://proceedings.mlr.press/v155/sharma21b.html)
  已覆盖 manipulation skill precondition learning。

**可能的窄差异：**

1. 目标 controller 是不微调的通用 VLA，不是单任务 grasp/locomotion policy；
2. semantic planner 只作 skill commitment，本地 governor 作逐步物理重观察；
3. 同时保留 success、failure、cost、uncertainty 和多层 outcome；
4. online input 受显式 deployment provenance firewall 约束；
5. 同一实现提供 Gate-0、controlled、Original Harness 和 full-agent 对照。

这些是有意义的组合/系统差异，但当前没有证据证明它们构成新的通用算法原理。

**Problem-Reality verdict：UNCERTAIN。** Gate-0 必须先显示：在相同 skill、checkpoint
和 reset group 中，pre-handoff state 对 frozen-VLA outcome 有稳定结构，且不是
perception/label artifact。

**Novelty verdict：UNCERTAIN（高 residual risk）。** 只有当控制严格、baseline
强、完整系统效应显著，并能具体指出 Harness planner heuristic 缺失的局部闭环能力
时，才可能形成较窄贡献。投稿前仍需围绕 2025–2026 frozen-VLA orchestration 做更广、
可复现的系统综述。

## E. 最终设计决定与证伪顺序

1. 保留 Phase A 为当前最强 in-scope 参考主线，因为它形成了完整、可执行、可证伪
   的实验系统，而不是因为 novelty 已成立。
2. 主方法暂称 outcome-calibrated myopic receding-horizon switching；不得称为已求解
   optimal stopping。
3. 实施工艺与证据完整性改进，不新增 Phase B research variant。
4. 经验顺序固定为：runtime readiness probe → tiny parity/Gate-0 → 三套独立 Gate-0
   landscape → group-held-out training/calibration/test → controlled baselines/ablations
   → full-agent comparison。
5. 若 Gate-0 不显示稳定 state-conditioned outcome structure，若 held-out calibration
   失败，或 fixed/threshold/projection 已达到相同 success-cost，应降级或放弃主张，
   而不是继续堆叠更复杂模型。
6. 若透明路由、checkpoint identity、reset identity、source/artifact identity 或 label
   定义任一未通过，则该 trial 不进入论文比较。

最终结论：**Phase B 没有找到一个比当前方案更强、同时现实且有清晰 residual
novelty 的新研究机制。** 下一步是严格按 Linux runbook 获取真实证据，再决定这条
主线是否值得形成论文方法；当前文档不声称任何经验优势。
