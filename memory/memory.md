# ScratchV engineering memory

[2026-10-14] 项目规范要求 pytest 测试类组织；新增测试使用 Test... 类，辅助函数保持模块级。迁移既有测试时保留参数化装饰器、fixture 参数和多行 DSL/汇编字符串；对比 AST 与去掉类名后的收集用例，防止整理时改变逻辑或漏收集。适用于测试组织调整。

[2026-10-14] LLVM 浮点常量按声明精度编码后用 bitcast 表示，避免十进制格式丢失有效位及加零丢失负零；显式 bit_pattern 必须保留并验证。LLVM 验证通过仍可能算错，回归需实际执行并核对返回位模式。LLVM 优化同样不能用宿主精度绕过 f32 输入舍入，适用于 typed DSL、ONNX 常量及显式 pass。

[2026-10-13] ONNX 浮点 Min/Max 的 NaN 传播不同于 RISC-V fmin/fmax 选择正常数字的规则，IR 必须保留来源要求，不能只按同名算子映射。用 ONNX 官方参考实现核对合法模型，再实际执行目标程序；同时检查 qNaN/sNaN 标志、普通数/无穷大和既有 DSL 行为。NaN 补偿运算只能在 NaN 路径执行，否则额外加法会引入溢出或 invalid；值参照工具也要遵循同一 IR 标记和目标符号零约定。

[2026-09-18] 课题 01 的解析成功和块数检查不能证明控制流执行正确；验收须核对比较契约、分支合流、循环变量更新与零次路径。当前 DSLInterpreter 的分支处理为空操作，不可作为控制流执行 oracle；未更新条件变量的旧 while 示例仅适合解析测试。

[2026-09-13] DSL benchmark 的详细日志按用例使用默认关闭的 details/summary 折叠，汇总表、失败检查和性能提示保留在外层；GitHub Markdown 与本地 HTML 均需支持，HTML 中的诊断源码必须转义。

[2026-09-13] DSL 错误标记缩进须按实际行号前缀的可见宽度计算，不能固定为 6；覆盖 9/10、99/100 位数边界及空格/tab、有色和无色输出，避免多位行号导致 caret 左移。ANSI 控制序列不计入宽度。

[2026-09-13] 维护者要求课题测试与 benchmark 接入原有 `.github/workflows/ci.yml` 的 test/benchmark jobs，不新建独立 pipeline；pytest tests/ 已自动发现 DSL 测试，报告复用现有 artifacts。适用于本仓库课题 PR 的 CI 接入。

[2026-09-13] DSL 诊断 benchmark 应用同一脚本和固定正确输入，在独立进程中导入基线与当前 checkout，并检查实际模块路径；同时比较源码与 IR 摘要。适用于课题 9 前端回归，避免 editable install 导致两侧都测到当前代码；错误输入的新增诊断能力单独验收，性能目标达标状态与功能通过状态分开报告。

[2026-09-19] 前端语义增强会有意改变固定 DSL 语料的 IR；基线 benchmark 应按用例显式列出允许变化，继续拒绝未列出的 IR 差异，并保留同语料、诊断正确与性能阈值检查。适用于多个前端课题共用同一 A/B benchmark 的 CI。

[2026-09-19] 课题专项 benchmark 应接入原有 CI job，摘要表常显，逐用例源码、IR 与汇编放入默认关闭的 details/summary，并同时产出 JSON/Markdown/HTML 到既有 artifact。适用于需要在 Actions Summary 展示详细编译日志的课题。

[2026-09-26] Topic 04 的 PassRegistry 注册只声明可用工厂，build 才按名称构造并调度；禁用先于构造。IR 和汇编分阶段建管线，常量折叠与汇编常量加载合并不可混用；新增调用点须使用统一工厂，避免旧 ConstantFolder(program).run() 接口。
