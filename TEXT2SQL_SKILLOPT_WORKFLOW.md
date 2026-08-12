Text-to-SQL Agent + SkillOpt 全流程运行说明
===========================================

本文档描述当前仓库中已经实现的 Train/Dev SkillOpt 流程。
当前分支：text-tosql-agent-skillopt

重要范围：
1. BIRD Train 用来生成轨迹并优化 skill。
2. Real Dev 用作 Gate（验证集），决定候选 skill 是否被接受。
3. BIRD Mini-dev 被保留为最终 Test，目前不参与训练和验证。
4. 当前阶段不运行 Mini-dev/Test。


一、数据集如何划分
==================

1. Train
--------

原始目录：

  ./bird_train_datas

原始题目文件：

  ./bird_train_datas/train.json

数量：9428题，涉及69个数据库。

每条记录自己的 SQL 字段就是 ground-truth SQL。预处理过程不会按照
train_gold.sql 的行号关联答案，避免行号错位。


2. Dev / Gate
-------------

Full Dev：

  ./bird_dev_datas/dev.json

Mini-dev：

  ./bird_mini_dev_datas/dev.json

Real Dev 的定义：

  Real Dev = Full Dev - Mini-dev 中全部500个 question_id

数量：

  Full Dev：1534题
  排除 Mini-dev：500题
  Real Dev：1034题

SkillOpt 不会把1034题全部用于每次验证，而是使用固定、可复现的 Gate
顺序：

  test.yaml：取固定顺序前50题
  pilot.yaml：取固定顺序前100题
  default.yaml：取固定顺序前200题

这50题是100题的子集，100题又是200题的子集。Gate 同时按数据库和难度
进行联合分层，随机种子为42。三个配置始终使用各自固定的题目集合，不会在
每次 Candidate 验证时重新抽样。


3. Test
-------

当前约定：

  ./bird_mini_dev_datas = 最终 Test

它不参与 skill 训练，不参与 Gate，也不在当前流程中执行。只有 skill
进化和选择完成后，才应该单独运行最终 Test。


二、两个模型分别做什么
======================

模型定义统一存放在仓库根目录：

  ./model_config.yaml

model_config.yaml 是模型注册表。每个 models 下的 key 是配置名称，例如：

  Qwen3.6-27B-no-think:
    base_url: http://.../v1
    model_name: "Qwen3.6-27B"
    api_key: empty
    temperature: 1.0
    ...

SkillOpt 本次运行使用哪两个模型，不在 model_config.yaml 里选择，而在：

  ./skillopt/configs/text2sql/default.yaml

对应配置：

  env:
    agent_model_config_name: Qwen3.6-27B-no-think
    optimizer_model_config_name: Qwen3.6-27B-no-think

两者含义：

1. agent_model_config_name
   Target Agent 模型。它读取 question + evidence + system prompt + learned
   skill，执行完整的多轮 Tool Call，最后返回 <FINAL_SQL>。

2. optimizer_model_config_name
   SkillOpt Optimizer 模型。它不直接做 BIRD 题，而是读取 Agent 的成功/
   失败轨迹，提出 skill 修改、合并修改并选择最终修改。

注意：上面填写的是 model_config.yaml 中 models 下的配置名称，不是 API
请求里的 model_name。


三、第一次运行前的一次性准备
============================

以下命令都从仓库根目录执行：

  cd /Users/xuancheng/Pycharm_Project/koenshen_text-to-sql-agent-skillopt


步骤1：生成 SkillOpt Train/Dev 数据
----------------------------------
cp -r ../koenshen_bird_evaluate/data_bird_train ./bird_train_datas
cp -r ../koenshen_bird_evaluate/data_dev ./bird_dev_datas
cp -r ../koenshen_bird_evaluate/data_mini_dev ./bird_mini_dev_datas
python skillopt/scripts/prepare_bird_skillopt_data.py

正常输出应包含：

  Train: 9428
  Dev:   1034
  Train 10/200/500/1000: database 覆盖统计
  Gate 50:  difficulty 分布
  Gate 100: difficulty 分布
  Gate 200: difficulty 分布

生成目录：

  ./skillopt/data/bird_text2sql

主要文件：

  public/train.jsonl
  public/dev.jsonl
  private/train_gold.jsonl
  private/dev_gold.jsonl
  selections/train_order.json
  selections/dev_gate_order.json
  reports/split_report.json
  manifest.json

public 文件不包含 ground-truth SQL；private 文件保存 ground-truth SQL。
二者始终通过稳定 ID 关联，不通过行号关联。


步骤2：生成69个 Train 数据库的 Agent schema
-------------------------------------------
mkdir topics/bird_train/schemas
python topics/bird_train/convert_bird_train_schema.py

输入：

  bird_train_datas/train_tables.json
  bird_train_datas/train_databases/<db_id>/<db_id>.sqlite

输出：

  topics/bird_train/schemas/<db_id>/database.json

正常情况下应生成69个数据库 schema。

如果原始数据、预处理结果和 schemas 都没有改变，这两个准备步骤不需要在
每次训练前重复执行。


四、推荐先运行小规模完整流程
============================

当前配置总览：

| 配置 | 用途 | Train | Gate | Epoch | Batch | Accumulation |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| test.yaml | 只检查完整管线 | 10 | 50 | 1 | 10 | 2 |
| pilot_epoch2.yaml | 低成本验证跨 epoch、Slow/Meta Update | 200 | 100 | 2 | 40 | 2 |
| pilot.yaml | 单 epoch 的较大 Train Pilot | 500 | 100 | 1 | 40 | 2 |
| default.yaml | 当前正式实验 | 1000 | 200 | 2 | 40 | 2 |

进入 vendored SkillOpt 目录：

  cd skillopt

运行：

  python scripts/train.py --config configs/text2sql/test.yaml

test.yaml 当前配置：

  epoch：1
  Train：10题
  batch size：10
  accumulation：2
  每次 Analyst minibatch：5条轨迹
  Gate：50题
  Agent workers：1
  Analyst workers：1
  Slow Update：关闭
  Meta Skill：关闭
  Test：关闭

一次 test.yaml 完整运行的主要 Agent 工作量：

  初始 skill Baseline Gate：50题
  Train rollout：             10题
  Candidate skill Gate：      50题
  合计：                     110次完整 Agent 流程

每次 Agent 流程内部还可能有多轮模型调用和工具调用，因此“50题 Gate”只
是比200题小，并不是几分钟可以结束的 smoke test。workers=1 时，一次完整
运行可能需要几十分钟或更久。


特别注意：Train 子集采用固定分层顺序
--------------------------------------

DataLoader 不再截取原始 Train 文件的前N条，而是读取：

  selections/train_order.json

该顺序使用 seed=42 按 db_id 分层生成。200题、500题和1000题前缀在数量
允许时保证覆盖全部69个 Train 数据库。四个 Train 前缀的关系是：

  Train 10 ⊂ Train 200 ⊂ Train 500 ⊂ Train 1000

test.yaml 的10题仍然只适合验证流程，不用于证明跨数据库泛化能力。


Pilot：正式实验前的低成本验证
--------------------------------

test.yaml 跑通后，建议先运行：

  python scripts/train.py --config configs/text2sql/pilot.yaml

pilot.yaml 当前配置：

  epoch：1
  Train：500题（覆盖全部69个 Train 数据库）
  batch size：40
  accumulation：2
  Gate：100题
  Agent workers：4
  Analyst workers：4
  Test：关闭

每个 Step 累计两个 microbatch 后生成一个 Candidate。由于500不能被80整除，
Text2SQL DataLoader 会将500题均匀分配到14个 microbatch，保证每个 epoch
恰好执行500道不同题目，不会重复补样。

Pilot 的主要 Agent 工作量：

  steps = ceil(500 / (40 × 2)) = 7
  Train Agent 流程：500次
  Baseline + Candidate Gate：(1 + 7) × 100 = 800次
  合计约：1,300次完整 Agent 流程


Epoch-2 Pilot：验证跨 epoch 流程
---------------------------------

如果主要目标是用较低成本确认 epoch=2、Slow Update 和 Meta Skill 可以完整
运行，使用：

  python scripts/train.py --config configs/text2sql/pilot_epoch2.yaml

pilot_epoch2.yaml 当前配置：

  epochs：2
  Train：200题（覆盖全部69个 Train 数据库）
  batch size：40
  accumulation：2
  Gate：100题
  Slow Update：同样必须通过100题 Gate
  Agent workers：4
  Analyst workers：4
  Test：关闭

每个 epoch：

  steps = ceil(200 / (40 × 2)) = 3
  microbatch：6个

两个 epochs 共6个 Candidate Step。Text2SQL DataLoader 会将每个 epoch 的
200题均匀分配到6个 microbatch，保证每个 epoch 恰好执行200道不同题目，
不会重复补样。

不含 Slow/Meta Update 时的主要 Agent 工作量：

  Train Agent 流程：200 × 2 = 400次
  Baseline + Candidate Gate：(1 + 6) × 100 = 700次
  基础合计：约1,100次完整 Agent 流程

第二个 epoch 结束时还会执行 Slow Update：同一批20道 Train 分别使用上一
epoch skill 和当前 skill 执行，并可能增加一次100题 Slow Update Gate。因此
完整运行通常约为1,240次 Agent 流程，另有 Optimizer 的反思、合并和 Meta
Skill 调用。


如果只想更快确认程序流程是否可走通
------------------------------------

不新增第三个配置文件，可以临时覆盖参数：

  python scripts/train.py \
    --config configs/text2sql/test.yaml \
    --cfg-options \
    evaluation.sel_env_num=5 \
    train.train_size=2 \
    train.batch_size=2 \
    gradient.minibatch_size=2

对应主要 Agent 工作量：

  Baseline Gate 5 + Train 2 + Candidate Gate 5 = 12次 Agent 流程

这只能验证程序、模型连接、轨迹反思、skill 更新和 Gate 是否连通，不能用来
判断 skill 的真实效果。


五、一次 SkillOpt Step 内部发生什么
====================================

启动后首先执行：

  BASELINE — evaluate initial skill on Selection set

这是用初始 skill 跑固定 Gate，得到 current_score/best_score。当前 initial.md
只有一条注释，语义上接近没有 learned skill。

随后每个训练 Step 包含6个阶段：


1/6 ROLLOUT
-----------

从 Train 取一个 batch。对每道题：

1. 读取 question 和 evidence。
2. 使用 eval.py 中已有的 build_bird_question() 构建用户输入。
3. 创建 Text-to-SQL Deep Agent。
4. 将当前 learned skill 追加到固定 system prompt 后。
5. Agent 多轮调用 sql_db_list_tables、sql_db_schema、sql_db_query 等工具。
6. LLM 停止 Tool Call，并返回 <FINAL_SQL>。
7. 复用 eval.py 中已有的 SQL 提取逻辑。
8. 立即计算 BIRD execution reward。


2/6 REFLECT
-----------

将 Train rollout 按 reward 分为：

  failure trajectories：hard=0
  success trajectories：hard=1

再按 minibatch_size 分组。每一组调用一次 Optimizer：

  失败组：分析为什么当前 skill 没有帮助 Agent 做对
  成功组：总结哪些行为值得写进 skill

Optimizer 返回候选 patch。


3/6 AGGREGATE
-------------

Optimizer 合并多个失败/成功 patch，去除冲突和重复，得到统一修改池。


4/6 SELECT
----------

根据 edit budget 选择最重要的修改。

test.yaml 当前 edit budget 为2，所以最多选择2条修改。


5/6 UPDATE
----------

将选中的 patch 应用到当前 skill，生成 candidate_skill.md。


6/6 EVALUATE / GATE
-------------------

使用 Candidate skill 在同一组固定 Gate 题目上重新执行完整 Agent 流程。

当前 Gate metric：

  hard accuracy = 正确题数 / Gate题数

如果 Candidate score 严格高于 current score，则 ACCEPT；否则 REJECT。
被 ACCEPT 且超过历史 best score 时，成为新的 best_skill.md。

注意：Gate 是验证集并参与 skill 选择，所以它不能再充当最终 Test。


六、BIRD Reward 如何计算
======================

每道题的 ground-truth SQL 来自原始 JSON 记录的 SQL 字段。

当前 reward 复制自官方 BIRD evaluator 的核心执行逻辑：

1. 在题目对应的同一个 SQLite 数据库执行 predicted SQL。
2. 执行 ground-truth SQL。
3. 分别 fetchall()。
4. 比较：

     set(predicted_rows) == set(ground_truth_rows)

5. 相等：hard=1，soft=1.0。
6. 不相等、SQL执行错误或超时：hard=0，soft=0.0。

常见状态：

  correct           两个执行结果一致
  result_mismatch   SQL可以执行，但结果不一致
  execution_error   SQL执行失败
  timeout           SQL执行超时
  empty_prediction  没有提取到预测 SQL

当前没有预先缓存 ground-truth 执行结果。每次 reward 都在同一运行过程中执行
predicted SQL 和 ground-truth SQL，避免序列化/反序列化导致结果格式变化。


七、如何阅读运行日志
==================

批次开始：

  [Text2SQL rollout] phase=candidate_gate epoch=1/2 step=3/26
  epoch_step=3/13 steps_after_current=23 items=200 database_groups=11 workers=4

Train rollout 还会显示当前 microbatch：

  [Text2SQL rollout] phase=train epoch=1/2 step=3/26 epoch_step=3/13
  batch=5/26 accum=1/2 steps_after_current=23 items=39 ...

数据库 Agent 初始化：

  [phase=... epoch=... step=... batch=...] [DB setup] START ...
  [phase=... epoch=... step=... batch=...] [DB setup] READY ...

单题开始：

  [phase=candidate_gate epoch=1/2 step=3/26 epoch_step=3/13 ...]
  [Question 1/200] START id=... split=dev db=... difficulty=...

单题结束：

  [phase=candidate_gate epoch=1/2 step=3/26 epoch_step=3/13 ...]
  [Question 1/200] DONE ... status=correct hard=1 turns=3 elapsed=11.6s
  progress=1/200 remaining_questions=199 running_accuracy=1.000

字段含义：

  status             BIRD execution 状态或 Agent 错误
  phase              当前处于 Baseline、Train、Candidate Gate 或 Slow Update
  epoch              当前 epoch / 总 epoch
  step               当前全局 SkillOpt Step / 总 Step
  epoch_step         当前 epoch 内的 Step / 本 epoch 总 Step
  batch              当前 Train microbatch / 本 epoch 总 microbatch
  accum              当前 Step 内累计到第几个 microbatch
  steps_after_current 当前 Step 完成后还剩几个 SkillOpt Step
  hard               当前题 reward（0或1）
  turns              Agent 中间步骤数量
  elapsed            当前题耗时
  progress           当前批次完成进度
  remaining_questions 当前 rollout 还剩多少题
  running_accuracy   当前批次截至目前的累计正确率

常见 phase：

  baseline_gate          初始 skill 在 Dev Gate 上的验证
  train                  Train trajectory 生成
  candidate_gate         当前 Candidate skill 的 Dev Gate
  slow_compare_previous  Slow Update 使用上一 epoch skill 跑 Train 样本
  slow_compare_current   Slow Update 使用当前 skill 跑同一批 Train 样本
  slow_update_gate       Slow Update Candidate 的 Dev Gate

注意：running_accuracy 只代表当前 rollout 已完成题目的准确率，不是最终准确率。
开启多个 workers 时，Question 的 START/DONE 会乱序；判断当前 rollout 的完成
程度应看 progress 和 remaining_questions，而不是最后出现的 Question 编号。

批次结束：

  [Text2SQL rollout] COMPLETE items=50/50 correct=... accuracy=... elapsed=...


以下警告当前通常不影响 Qwen 运行：

  CLAUDE_API_KEY is not set
  SENSETIME_API_KEY is not set
  NOVA_API_KEY is not set

原因是 model_config.yaml 加载时检查了未被本次运行选择的其他模型。

以下信息表示可选实体检索未启用，不是程序退出：

  Entity retrieval: token index not found, skipping

当前 schemas 没有 entity_index.json，所以 Agent 跳过实体索引，继续使用
schema 和 SQL 工具。

如果一题在0.0秒内出现同一个 Agent TypeError，通常是模型配置或框架兼容
问题，不应把它理解为 SQL 错误。当前代码会对 unexpected keyword argument
类型的模型配置错误快速终止，而不是把整批题错误记成0分。


八、输出目录和关键文件
======================

每次运行会创建：

  ./skillopt/outputs/skillopt_text2sql_<API模型名>_<时间戳>

重要文件：

  config.json
    本次运行最终解析后的完整配置。

  summary.json
    Baseline、best score、接受/拒绝次数、Optimizer token统计等。

  history.json
    每个 Step 的历史记录。

  best_skill.md
    当前运行最终选择出的最佳 skill。这是最重要的产物。

  skills/skill_v0000.md
    初始 skill。

  skills/skill_vXXXX.md
    每个 Step 保存的 skill 状态。

  selection_eval_baseline/predictions/<question_id>/
    初始 skill 的 Gate 轨迹和执行结果。

  steps/step_XXXX/rollout/predictions/<question_id>/
    Train 轨迹。

  steps/step_XXXX/patches/
    Analyst 为失败/成功轨迹生成的 patch。

  steps/step_XXXX/merged_patch.json
    合并后的修改池。

  steps/step_XXXX/ranked_edits.json
    最终被选择的修改。

  steps/step_XXXX/candidate_skill.md
    本 Step 的候选 skill。

  steps/step_XXXX/selection_eval/predictions/<question_id>/
    Candidate Gate 轨迹和执行结果。

每道 prediction 通常包含：

  conversation.json
  evaluation.json
  target_system_prompt.txt
  target_user_prompt.txt

summary.json 中的 token_summary 当前统计的是 SkillOpt Optimizer 调用，不包含
Target Agent 全部模型 token，因此不能把它当成整个实验的总 token 成本。


九、完整配置如何运行
====================

命令：

  cd skillopt
  python scripts/train.py --config configs/text2sql/default.yaml

default.yaml 当前配置：

  epochs：2
  train_size：1000（固定分层 Train 顺序的前1000题）
  batch_size：40
  accumulation：2
  Gate：200题
  Slow Update：同样必须通过200题 Gate
  Agent workers：4
  Analyst workers：4

每个 epoch 的更新规模：

  每个 epoch steps = ceil(1000 / (40 × 2)) = 13
  2个 epochs 总 steps = 26
  每个 Step 都会执行一次200题 Candidate Gate

大约会产生：

  Train Agent 流程：1000 × 2 = 2,000次
  Candidate Gate Agent 流程：26 × 200 = 5,200次
  初始 Baseline Gate：200次
  基础合计：约7,400次完整 Agent 流程

Text2SQL DataLoader 会将1000题均匀分配到26个 microbatch，因此每个 epoch
恰好执行1000道不同题目，不会为了填满最后一个 accumulation step 重复补样。
第二个 epoch 的 slow update 还可能增加少量对比 rollout 和一次200题 Gate；
以上也不包括 Optimizer 的 Analyst、Merge 和 Ranking 调用。


十、结果可信度注意事项
====================

1. Gate 随机性
--------------

如果 Target Agent 的 temperature 大于0，同一个 skill 在相同50题上重复运行，
正确率也可能变化。例如一次48%，另一次52%。Candidate 比 Baseline 高几个点
不一定全部来自 skill。

要提高 Gate 决策可信度，可以考虑：

  - Gate 时使用更低 temperature；
  - 增大 Gate 题数；
  - 对同一 skill 重复评测后取平均；
  - 对 Baseline/Candidate 使用配对、可复现采样（如果服务端支持）。


2. 基础设施错误不能等同于 SQL 错误
----------------------------------

网络失败、模型参数错误、Middleware 错误都不是 Agent 的 Text-to-SQL 能力
错误。此类题目不应该直接作为 hard=0 影响 Gate，理想处理方式是重试，仍失败
则让当前 Gate 无效并终止，而不是接受一个受系统故障污染的 score。


3. Train 抽样应覆盖多个数据库
-----------------------------

当前 Train 顺序使用 seed=42 按 db_id 固定分层。Train 200、Train 500 和
Train 1000 都覆盖全部69个 Train 数据库，避免直接截取原始 train.json 前N条
导致数据库集中。BIRD Train 本身没有 difficulty 字段，因此 Train 只能按
数据库分层，不能像 Dev Gate 一样按 db_id + difficulty 联合分层。


4. 与 eval.py 的当前差异
-----------------------

SkillOpt rollout 与 eval.py 共用：

  build_bird_question
  extract_final_answer
  extract_intermediate_steps
  extract_sql_from_bird_answer
  extract_sql_from_steps
  create_sql_deep_agent

但普通 BIRD eval.py 在 bird_mini_dev topic 下还会自动使用存在的
few_shot_index，并默认读取 focused_schema.json。当前 SkillOpt rollout 没有注入
Few-shot；Real Dev Gate 的 question_id 又不在 Mini-dev focused_schema.json 中。

如果最终部署/测试命令依赖 Few-shot，那么训练和 Gate 也应对齐相同的 Few-shot
逻辑，否则训练时 Agent 输入与最终测试时输入不完全一致。


5. 与持续更新的 Agent 框架之间的边界
---------------------------------------

SkillOpt 始终调用项目当前的 agent.py，不维护静态 Agent 副本。agent.py 中只
保留两个默认值为 None 的 SkillOpt 专属接口：

  skillopt_skill_content
  skillopt_database_root_override

纯 Agent 的 eval.py 不传这两个参数，因此不加载 SkillOpt、不附加 skill，也不
覆盖 topic 自己的数据库路径。

skillopt/skillopt/envs/text2sql/agent_bridge.py 只集中当前 agent.py、eval.py 和
question context 的版本敏感调用，不参与 Train/Dev 数据选择、BIRD reward、
Analyst、Reflect、Aggregate、Select 或 Gate。Text2SQLAdapter.setup() 会在任何
模型请求和 SQL 执行之前进行只读兼容性检查；缺失接口时立即终止。

整个 topics/ 目录属于 Text-to-SQL Agent 和开发者提供的运行资源。SkillOpt 只
读取 bird_train 和 bird_mini_dev，不移动、不生成也不重写其中内容。


十一、一次推荐的实际操作顺序
============================

1. 确认位于分支：

     git branch --show-current

   应输出：

     text-tosql-agent-skillopt

2. 确认原始数据目录存在：

     bird_train_datas
     bird_dev_datas
     bird_mini_dev_datas

3. 首次运行或数据变化后执行：

     python skillopt/scripts/prepare_bird_skillopt_data.py
     python topics/bird_train/convert_bird_train_schema.py

4. 在 model_config.yaml 的 models 下定义需要的模型。

5. 在 skillopt/configs/text2sql/default.yaml 中选择 Agent 和 Optimizer 模型。

6. 先运行12次 Agent 的临时极小流程，确认连接和程序正确。

7. 再运行 test.yaml 的10 Train + 50 Gate 完整流程。

8. 查看输出目录中的 summary.json、ranked_edits.json 和 best_skill.md。

9. 运行 pilot.yaml 的500 Train + 100 Gate、1 epoch 实验。

10. 检查 Gate 提升是否超过同一 Baseline 多次运行的随机波动，并检查是否存在
   agent_error、execution_error 或 timeout。

11. 需要先低成本验证跨 epoch 逻辑时，运行 pilot_epoch2.yaml 的200 Train +
    100 Gate、2 epochs 实验。

12. Pilot 结果正常且预算允许时，再运行 default.yaml 的1000 Train + 200
    Gate、2 epochs 实验。

13. Train/Dev 流程稳定并选出最终 skill 后，再单独设计 Mini-dev/Test 评测；
    不要在 SkillOpt 训练过程中使用 Test gold。
