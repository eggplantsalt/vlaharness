# Controller-Handoff Autonomous Research Review

> 审查日期：2026-08-08。本文记录的是完成实现后的研究判断，不是实验结果。
> 结论中的“实现事实”来自当前 checkout 的静态检查；“文献事实”来自原论文；
> “推断”与“待验证”均不应写成 established finding。

## A. Current-design critique

### 作者视角

Phase-A 的最强价值不是某个单独学习器，而是把一个可证伪的问题完整接到
真实执行边界上：冻结 VLA、部署可用观测、逐步分析控制、每步重新观察、真实
成功与失败结果、显式成本/不确定性、无 privileged online input，并同时保留
Original Harness、控制器级和完整 agent 级比较。

实现中的主策略在每次观察后计算：当前状态立即 handoff 的成本，以及若干
target-relative candidate 的“stage 一步后再 handoff”成本；若未来 candidate
更优，则只执行一个有界分析动作，重新观察真实状态，再重新决策。它没有在
预测状态上连续开环执行。这一工程结构是合理的，也能直接检验局部 governor
是否优于 LLM 反复微调。

但作者必须主动收窄方法表述：当前算法不是已经求解的 optimal stopping、POMDP
或多步最优控制。它没有 staging transition model、observation model、长期
value-to-go，也没有对未来视觉状态作可信生成；视觉量只能保持当前值或显式记为
unavailable/approximated。准确名称应是 **outcome-calibrated myopic
receding-horizon switching**（结果校准的单步滚动切换），除非以后用形式化模型与
实验支持更强说法。

### 最苛刻 reviewer 视角

1. **核心机制与既有工作高度重叠。** 学习目标控制器的成功概率/可达域、寻找
   合适起始状态、训练 setup/transition policy、以及用真实后继任务成败学习二元
   switch，都已有直接先例。不能把“learn competence and switch”本身声明为
   novelty。
2. **Phase-A 的“optimal”目前只是配置成本下的局部 argmin。** staging 成本是
   几何代理，failure/VLA/hysteresis 权重由实验配置给出；不同权重可能改变全部
   结论。必须做 sensitivity 和固定权重协议。
3. **future-state 估计受限。** 未来 EEF 几何是构造的，未来 perception、遮挡、
   接触与 reachability 不是。保守 bootstrap 分数是经验不确定性，不是安全保证。
4. **只在 handoff 后得到 VLA outcome。** Gate-0 的受控干预覆盖减轻选择偏差，
   但在线 governor 诱导的状态分布、candidate lattice 与 Gate-0 分布仍可能不同；
   calibration 必须在独立 task/reset/group 上验证。
5. **标签层级可能改变结论。** primitive success、skill success、official task
   termination、truncation、planner finish、RPC/VLA failure 不能互换。主 claim 必须
   预先固定 target label，并报告其他信号的一致性与缺失率。
6. **系统改进不等于算法贡献。** 更少 LLM turns、可恢复日志、配置隔离和严格
   provenance 很重要，但属于系统/实验严谨性，不能替代机制新颖性。
7. **尚无 runtime 或 empirical evidence。** 当前 Windows 阶段没有运行测试、
   simulator、Gate-0、模型推理或 GPU。实现完成只说明实验现在可执行。

### 最关键的研究判断

Phase-A 的问题现实性是 **UNCERTAIN**：Harness VLA 的真实 perturbation 和
restaging 问题提供了合理动机，但“pre-handoff 连续状态是否产生足够强、可迁移、
可校准的 outcome gradient”仍必须由 Gate-0 证明。其机制新颖性同样是
**UNCERTAIN**，不是 PASS；残余空间只能来自非常具体的系统组合与实证差异，不能
来自通用 competence/switching 叙述。

## B. Explored alternatives

### 1. Conformal risk-controlled handoff — rejected as a new main mechanism

- **Problem-Reality：PASS。** 当前 bootstrap/quantile score 没有 finite-sample
  risk guarantee；如果论文声称安全或可控风险，这是真问题。
- **Novelty：FAIL。** [Conformal Decision Theory（2023-10-09）](https://arxiv.org/abs/2310.05921)
  已把“信任 nominal policy 还是切换到 safe backup”作为直接应用，并校准决策风险；
  [Conformal Policy Learning（2023-11-02）](https://arxiv.org/abs/2311.01457)
  更直接用 conformal quantile 在不同 base policies 间切换并给出形式保证。
- **决定：** 不实现为 Phase-B 新 variant。未来若真实 calibration/risk-coverage
  结果暴露问题，可作为已有方法导出的 baseline/extension，不能包装成核心新意。

### 2. Information-gathering / active-perception staging — rejected

- **Problem-Reality：UNCERTAIN。** 当前 target uncertainty/occlusion 可能确实让
  “move to see, not move to handoff”有价值，但服务器 probe 和 Gate-0 尚未证明这在
  本系统中是主要失败源。
- **Novelty：FAIL / 高风险。** 主动改变视角以获取 manipulation-relevant 信息是
  成熟方向；[Active Perception and Representation for Robotic Manipulation
  （2020-03-15）](https://arxiv.org/abs/2003.06734) 已用 viewpoint changes 做定位与
  manipulation，[Real-World Reinforcement Learning of Active Perception
  Behaviors（2025-12-01）](https://arxiv.org/abs/2512.01188) 进一步在真实 manipulation
  中学习信息收集行为。
- **决定：** 当前实现需要额外 observation dynamics/world model 或专门 active
  perception policy，既有重叠高且 scope drift；不污染稳定 Phase-A 路径。

### 3. Causal/off-policy counterfactual handoff value — rejected

- **Problem-Reality：UNCERTAIN。** 只观察已选择 handoff 的结果会产生 selection
  bias，但 Gate-0 已通过受控 candidate intervention 收集正负结果，在线 trial 也可
  随机化/保留 propensity。现阶段尚无证据说明更复杂识别方法是必要瓶颈。
- **Novelty：UNCERTAIN，收益不足。** 通用 off-policy evaluation、causal policy
  learning 与 doubly robust 估计已经成熟；在没有 overlap/ignorability 证据时，换
  一个 estimator 不会自动识别未执行 counterfactual。
- **决定：** 先用受控数据、严格 group split、held-out calibration 和 coverage
  diagnostics。若以后日志策略改变，记录 propensity 是合理工程扩展，但不是当前
  主机制。

### 4. Learned setup/transition policy or RoA planner — rejected

- **Problem-Reality：UNCERTAIN。** 脚本 servo 是否覆盖不足要等真实 staging
  failure/碰撞数据；不能因为更复杂方法存在就假定问题成立。
- **Novelty：FAIL。** [Learning Setup Policies（2021-01-23；RA-L 2022）](https://arxiv.org/abs/2101.09391)
  明确学习连接 pretrained controllers 的 setup policy；[Training Transition
  Policies via Distribution Matching（2021-10-08；ICLR 2022）](https://arxiv.org/abs/2110.04357)
  用二元 stay/switch 与后继任务真实成功/失败 reward 学切换；[Hybrid Systems Neural
  Control with Region-of-Attraction Planner（L4DC 2023）](https://proceedings.mlr.press/v211/meng23a.html)
  学习跨 mode RoA 并规划落入下一 mode 的可达域。
- **决定：** 与现有机制过近，并会把“简单分析 staging + 冻结 VLA”扩大为新控制器
  学习项目；不保留为 Phase-B candidate。

### 5. Multi-step world-model / POMDP optimal stopping — rejected for this phase

- **Problem-Reality：PASS（formulation gap），effect size UNCERTAIN。** 它确实能
  解决当前方法没有 value-to-go 的形式缺口，但尚无证据说明单步重观察不足。
- **Novelty：UNCERTAIN，scope drift：FAIL。** 完整 world model/POMDP 会改变论文
  目标、数据要求和实现边界，也违反本阶段明确 non-goal。
- **决定：** 把当前方法命名得更诚实；不为了更漂亮的数学叙述引入 speculative
  model。

### 6. Absorbed engineering improvements

以下改变提高可执行性与证据质量，但不作为 research-method novelty：

- direct frozen-VLA baseline 不再需要 target/perception；
- 科学 controller config hash 与完整 run config hash 分离；
- live suite/task/seed/reset identity 交叉验证，reset 缺失时 fail closed；
- 每个 trial 强绑定自己的 model artifact，配置内相对路径有固定解析基准；
- staging、perception、cancellation、RPC/VLA、termination/truncation 分开记账；
- VLA 进入后取消仍保留 invocation attempt/time，未知 chunk/action 不伪造为零；
- 无 future candidate、single-class split、feature/artifact mismatch 明确失败；
- 保存的 split assignment 可直接 materialize 为精确 held-out JSONL；
- runtime probe 可在明确隔离/重置确认后捕获真实 VLA observation 并验证实际
  action shape；
- dry-run、进程隔离、resume lifecycle、manifest/checksum 与输出 containment
  形成一条可审计执行路径。

## C. Prior-art / novelty audit

### Surviving candidate: narrowed Phase-A reference hypothesis

这里“surviving”只表示值得做实验，不表示已通过 novelty PASS。

**机制本质。** 在 agent 已语义承诺一个 contact-rich skill 后，本地 governor 用
当前部署观测和真实正负 VLA outcome 模型，反复比较“现在把控制交给冻结 VLA”与
“付出一次有界分析 staging 成本后再评估”；每次 staging 后重新观察，禁止在线
privileged state。

**Already-grounded problem。** [Harness VLA（2026-07-09）](https://arxiv.org/abs/2607.08448)
本身把 frozen VLA 暴露为可重试 contact-rich primitive，并由 agent 做 analytic
staging/restaging；这说明系统中确实存在调用条件和 operating-range 问题。但它并不
自动证明连续局部 handoff governor 优于 planner heuristic。

**最强近邻与实质重叠。** 

- [Learning When to Switch（2020-11-01；IROS 2021）](https://arxiv.org/abs/2011.00440)
  为目标 controller 学成功切换概率并据此选择 switch region；它已覆盖“continuous
  state + learned success estimator + threshold switching”。
- [Hierarchical Policies for Cluttered-Scene Grasping with Latent Plans
  （2021-07-04；RA-L 2022）](https://arxiv.org/abs/2107.01518) 学 option classifier，
  判断当前状态切到预训练 grasp policy 是否会成功；它已覆盖 manipulation 中的
  outcome-trained controller switch。
- [Training Transition Policies via Distribution Matching
  （2021-10-08；ICLR 2022）](https://arxiv.org/abs/2110.04357) 的二元 switch 决策直接
  使用后继 pretrained policy 的成功/失败作为 reward；它覆盖 transition/setup 与
  成败驱动切换。
- [Where To Start?（CoRL 2022 / PMLR 2023）](https://proceedings.mlr.press/v205/vosylius23a.html)
  学习技能从某机器人配置开始是否能成功/无碰撞，并在 deployment 优化起始配置，
  再运动规划到那里执行既有技能；它与 competence projection/start-state
  optimization 非常接近。
- [Relational Learning for Skill Preconditions（CoRL 2020 / PMLR 2021）](https://proceedings.mlr.press/v155/sharma21b.html)
  已覆盖 manipulation skill precondition learning，进一步压缩“学习可执行域”作为
  独立新意的空间。

**可能的实质差异。** 当前候选把上述思想放到一个特定而严格的系统边界：

1. 目标 controller 是不微调的通用 VLA，而非单任务 locomotion/grasp policy；
2. semantic planner 只做 skill commitment，本地 governor 做逐步物理重观察；
3. 同时保留 success、failure、cost、uncertainty 与多个层级 outcome；
4. online policy 输入明确受 deployment provenance firewall 约束；
5. 同一实现提供 Gate-0、controller-only、Original Harness 与 full-agent 对照。

这些是有意义的组合/系统差异，但目前没有证据证明它们构成新的通用算法原理。

**Problem-Reality verdict：UNCERTAIN。** Gate-0 必须先显示：在相同 skill、checkpoint
和 reset group 下，pre-handoff state 对 frozen-VLA outcome 有稳定结构，且不是
perception/label artifact。

**Novelty verdict：UNCERTAIN（高 residual risk）。** 最危险的 reviewer 解释是：
“这是既有 success-based switching / skill precondition / start-state optimization，
换成 VLA 并加工程系统。”只有在控制严格、baseline 强、完整系统效应显著、并能
明确指出 Harness planner heuristic 做不到的局部闭环能力时，才可能形成较窄的
贡献。投稿前还需要围绕 2025–2026 frozen-VLA orchestration 做一次更广、可复现的
系统综述。

### No surviving new Phase-B research variant

Conformal switching、active-perception staging、causal/OPE、learned setup/RoA、
multi-step world model 都没有同时通过 Problem-Reality 与 Novelty 两道门。因此本轮
没有新增 speculative policy/config，也没有伪造一个“更高级”机制来制造产出。

## D. Final design decision

1. **保留 Phase-A 为当前最强 in-scope 参考主线。** 原因是它已形成完整、可执行、
   可证伪的实验系统，而不是因为其 novelty 已成立。
2. **收窄 claim。** 主方法暂称 outcome-calibrated myopic receding-horizon
   switching；不能称为已求解 optimal stopping。
3. **实施工程 refinement，不实施新 research variant。** 上述身份、路径、reset、
   split materialization、runtime capture、失败/取消记账等改进直接提高科学有效性，
   但不改变 Phase-A reference method。
4. **经验决策顺序固定。** runtime probe → tiny parity/Gate-0 → Gate-0 landscape →
   group-held-out calibration → controlled baselines/ablations → full-agent comparison。
5. **允许被证伪。** 如果 Gate-0 没有稳定 state-conditioned outcome structure，或者
   fixed/threshold/projection 已达到同等 success-cost，应该放弃/降级主张，而不是
   再堆叠一个更复杂模型。

最终结论很直接：**Phase B 没有找到一个比当前方案更强、同时现实且具清晰 residual
novelty 的新研究机制。** 当前最合理的下一步不是继续设计模块，而是在 Linux GPU
服务器按 runbook 获取真实证据，再决定这条主线是否值得成为论文方法。
