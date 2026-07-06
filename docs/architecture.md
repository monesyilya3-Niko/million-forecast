# 系统架构

## 部署决策

采用“本地服务 + 浏览器界面”：

```text
本机浏览器
    ↓ 127.0.0.1:8502
Streamlit 界面层
    ↓
应用服务层
    ↓
特征层 / 模型层 / 回测层
    ↓
DuckDB + Parquet + 模型文件
    ↓
D:\football-model
```

这不是云端网页。关闭本地服务后，其他设备不能访问。后续可使用 Tauri、WebView2 或 PyInstaller 启动器封装成桌面窗口，而无需重写数据和模型层。

## 模块边界

```text
src/football_model/
├─ core/       配置、路径和运行参数
├─ data/       DuckDB、仓库和外部数据适配器
├─ features/   严格基于 cutoff_at 的特征生成
├─ models/     统计模型和机器学习模型接口
├─ services/   单场、批量、训练、预测和回测用例
├─ ui/         Streamlit组件与页面
└─ engine.py   当前Dixon-Coles概率基线
```

## 数据存储

- DuckDB：比赛索引、赔率快照、预测记录和模型注册表。
- Sporttery实时层：官方赛程、销售状态、玩法和SP，每60秒写入一次可去重快照。
- Parquet：规模较大的赛果、事件、xG和特征明细。
- artifacts：训练后的模型和校准器。
- reports：回测报告与模型评估结果。

## 下一阶段

1. 接入中国体彩官方SP快照或合法授权数据源。
2. 建立球队名称、赛事和时区标准化。
3. 对已训练Dixon-Coles模型执行Walk-forward样本外回测。
4. 增加Elo、XGBoost残差和概率校准。
5. 建立模型版本晋级与自动监控流程。
