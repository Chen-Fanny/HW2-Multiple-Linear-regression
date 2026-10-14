# HW2: 共享單車需求預測 (Bike Sharing Demand Prediction)
## 多元線性迴歸 (Multiple Linear Regression) 與特徵選取 (Feature Selection)

---

### 學生與課程資訊 (Student Information)
- **課程名稱**：物聯網數據分析與應用
- **學生學號**：`7115056046`
- **學生姓名**：`陳奕帆`
- **GitHub 專案儲存庫**：[https://github.com/Chen-Fanny/HW2-Multiple-Linear-regression.git](https://github.com/Chen-Fanny/HW2-Multiple-Linear-regression.git)
- **GitHub Pages 線上互動報告**：[https://chen-fanny.github.io/HW2-Multiple-Linear-regression/](https://chen-fanny.github.io/HW2-Multiple-Linear-regression/)

---

### 1. 專案概述 (Project Overview)
本研究以智慧城市與物聯網交通微型移動（Micromobility）為核心應用情境，針對美國華盛頓特區 **Capital Bikeshare** 系統之每小時共享單車租借需求進行精準預測。本專案透過標準的 **CRISP-DM** 六大階段，建構多元線性迴歸 (Multiple Linear Regression) 基準模型，比較五種特徵選取演算法，深入診斷多重共線性 (Multicollinearity / VIF)，並引入具統計理論覆蓋保證之 **獨立校準集分割共形預測 (Split Conformal Prediction)**，產出無分佈假設的 95% 不確定性需求區間。

- **資料集來源**：[Kaggle Bike Sharing Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset) / UCI Machine Learning Repository (`hour.csv`)
- **預測目標變數**：`cnt` (每小時總借車需求數量)
- **核心執行程式**：`7115056046_hw2.py` (作業主程式) 與 `main.py`
- **互動網頁展示**：`index.html` (已部署至 GitHub Pages)

---

### 2. 專案目錄架構 (Project Structure)
```
HW2-Multiple-Linear-regression/
├── 7115056046_hw2.py                     # 作業核心執行腳本 (CRISP-DM 完整管線)
├── main.py                               # 專案主執行程式
├── requirements.txt                      # 專案相依套件清單
├── .gitignore                            # Git 版本控制忽略清單
├── README.md                             # 專案完整技術說明文件
├── index.html                            # GitHub Pages 互動展示主頁 (含 OLS / VIF / 模擬器)
├── style.css                             # 現代深淺色自適應樣式表
├── app.js                                # 前端即時矩陣推論與共形區間計算邏輯
├── web_server.py                         # 本地輕量 HTTP 伺服器
├── data/                                 # 原始與處理資料集
│   ├── hour.csv                          # 每小時單車需求核心數據 (17,379 筆)
│   ├── day.csv                           # 每日匯總數據 (731 筆)
│   └── Readme.txt                        # UCI 原始資料集欄位說明文件
├── results/                              # 評估報表與高品質統計視覺化圖表
│   ├── baseline_linear_regression_metrics.csv # 基準模型各切分集指標
│   ├── ols_detailed_coefficients_vif.csv      # 完整迴歸係數與 VIF 共線性診斷表
│   ├── feature_selection_comparison_all.csv   # 5 種選取方法在所有 k 階子集表現
│   ├── best_models_per_method.csv             # 各方法最優特徵數與驗證集效能
│   ├── feature_coefficients.csv               # 最佳特徵子集之標準化迴歸係數
│   ├── test_prediction_intervals.csv          # 測試集實際值、預測值與 95% 預測上下限
│   ├── eda_target_distribution.png            # 目標變數分佈與箱形圖
│   ├── eda_hourly_demand.png                  # 工作日 vs 假日雙峰需求曲線
│   ├── eda_seasonal_weather.png               # 季節與天候狀態需求差異
│   ├── eda_correlation_heatmap.png            # Pearson 相關係數熱圖 (顯示 temp-atemp 高相關)
│   ├── feature_selection_rmse_vs_k.png        # 特徵數 k vs 驗證集 RMSE 變化圖
│   ├── feature_selection_r2_vs_k.png          # 特徵數 k vs 驗證集 R² 變化圖
│   ├── train_vs_val_performance.png           # 訓練集 vs 驗證集 RMSE 泛化診斷圖
│   ├── prediction_interval_95.png             # 測試集 168 小時 95% 預測區間時序圖
│   ├── actual_vs_predicted_scatter.png        # 測試集實際值 vs 預測值散佈圖 (y = x 參考線)
│   ├── residual_distribution.png              # 殘差分佈與常態分佈擬合圖
│   ├── residuals_vs_predicted.png             # 殘差 vs 預測值 (異質變異檢驗圖)
│   └── feature_importance_coefficients.png    # 特徵重要性長條圖 (標準化 Beta)
└── models/                               # 序列化持久化模型
    ├── baseline_linear_regression.joblib      # 全特徵多元線性迴歸模型
    ├── best_linear_regression.joblib          # 最優特徵子集線性迴歸模型
    └── feature_scaler.joblib                  # 訓練集擬合之標準化轉換器
```

---

### 3. CRISP-DM 六大階段實作細節

#### 階段一：業務理解 (Business Understanding)
- **物聯網痛點**：都市共享單車具強烈潮汐通勤效應與天氣敏感性，供需失衡將直接導致站點無車可借（痛失營收）或車位已滿無處可還（引發民怨），且調度卡車派遣成本高昂。
- **目標**：以具高物理可解釋性之多元線性迴歸模型預測每小時總租借量 `cnt`，並提供統計學保證之 95% 不確定性區間，作為站點車輛動態調配與備車決策支援。

#### 階段二：資料理解 (Data Understanding & EDA)
- 資料筆數：17,379 筆每小時紀錄（2011/01/01 至 2012/12/31），數據完整無缺失值。
- 共線性發現：氣溫 `temp` 與體感溫度 `atemp` 相關係數高達 **0.9877**，呈現極度嚴重之共線性 (Multicollinearity)。
- 需求模式：工作日呈現顯著「晨尖峰 (08:00-09:00)」與「傍尖峰 (17:00-18:00)」之雙峰分佈；週末與假日則呈現平緩午後單峰。

#### 階段三：資料準備與防範外洩 (Data Preparation & Leakage Prevention)
1. **防止目標外洩 (Target Leakage)**：
   - 強制剔除 `casual`（散客）與 `registered`（會員），因兩者相加恆等於目標 $cnt = casual + registered$。
2. **剔除純流水號**：剔除 `instant` 避免模型記憶無效的遞增索引。
3. **防止時序穿越外洩 (Data Leakage)**：
   - 拒絕隨機打亂 (Random Shuffling)，嚴格依時間序列遞增順序切分：
     - **訓練集 (Train)**：前 60.0% (10,427 筆, 2011-01-01 至 2012-03-15)
     - **驗證集 (Validation)**：次 15.0% (2,606 筆, 2012-03-15 至 2012-07-02) — 用於特徵選取與挑選最優特徵數 $k^*$
     - **校準集 (Calibration)**：次 10.0% (1,737 筆, 2012-07-02 至 2012-09-12) — 專屬分割共形預測殘差校準
     - **測試集 (Test)**：末 15.0% (2,609 筆, 2012-09-12 至 2012-12-31) — 獨立未接觸之未來測試集
4. **特徵工程 (共構建 18 個輸入特徵，符合 10~20 個特徵要求)**：
   - 行事曆趨勢：`yr` (年份趨勢)、`holiday` (國定假日)、`workingday` (工作日)
   - 氣象數值：`temp` (氣溫)、`atemp` (體感溫度)、`hum` (相對濕度)、`windspeed` (風速)
   - 季節虛擬變數：`season_summer`、`season_fall`、`season_winter` (以春季為基底)
   - 天候虛擬變數：`weather_mist`、`weather_bad` (以晴天少雲為基底)
   - 週期諧波特徵：24 小時日週期諧波 `hr_sin`、`hr_cos`；12 小時通勤半日諧波 `hr_sin2`、`hr_cos2`
   - 物理交互項：工作日尖峰通勤指示器 `is_rush_hour`、溫濕度交互項 `temp_hum`

#### 階段四：多元線性迴歸與五種特徵選取 (Modeling & Feature Selection)
本專案實作並橫向比較五種特徵選取方法，並以未正則化之 `LinearRegression` 作為子集預測模型：
1. **Lasso (L1 Regularization)**：以 5 折交叉驗證自適應決定懲罰強度，自動將嚴重共線之 `atemp` 係數強力壓縮歸零。
2. **Forward Sequential Feature Selection (Forward SFS)**：前向逐步貪婪加入使驗證集 RMSE 下降最多之特徵。
3. **Recursive Feature Elimination (RFE)**：以線性模型係數絕對值為準則，遞迴剔除貢獻度最小之特徵。
4. **ElasticNet (L1 + L2 Regularization)**：兼具 L1 稀疏性與 L2 分組效應，同時保留並分攤共線之 `temp` 與 `atemp`。
5. **Exhaustive Feature Search (有限計算界限窮舉搜尋, $k=1 \dots 6$)**：向量化法方程式加速，在 3 秒內遍歷 31,179 種組合，求出全域數學最優 6 特徵為 `[yr, atemp, hr_sin, hr_cos, is_rush_hour, temp_hum]`。

#### 階段五：模型評估與 95% 共形預測區間 (Evaluation & Split Conformal Prediction)
- 統計原理：在獨立校準集上計算絕對非一致性殘差 $s_i = |y_{cal, i} - \hat{y}_{cal, i}|$。
- 95% 共形臨界分位數：$q = \text{Quantile}_{\lceil (n+1)(1-\alpha)\rceil / n}(\{s_i\}) = \mathbf{318.81}$ 輛。
- 測試集預測區間：$[ \max(0, \hat{y}_t - q), \hat{y}_t + q ]$。
- 未來測試集實證覆蓋率達 **94.40%**（名義水準 95.0%），平均區間寬度 525.99 輛。

#### 階段六：部署與推論效能 (Deployment)
- 矩陣推論延遲小於 0.05 毫秒，極速輕量，適宜部署於邊緣 IoT 設備或雲端容器中。
- 提供上線監控機制、滑動視窗再校準機制與概念漂移檢測機制。

---

### 4. 關鍵實驗指標總覽 (Key Evaluation Metrics)

| 評估階段 | 評估資料集 | 樣本數 ($n$) | 均方誤差 (MSE) | 均方根誤差 (RMSE) | 判定係數 ($R^2$) | 調整後 $R^2$ | 平均絕對誤差 (MAE) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **訓練集 (Train)** | 2011/01 ~ 2012/03 | 10,427 | 5,452.68 | 73.84 輛 | **0.6991** | 0.6986 | 55.62 輛 |
| **驗證集 (Validation)** | 2012/03 ~ 2012/07 | 2,606 | 17,882.13 | 133.72 輛 | **0.6131** | 0.6104 | 102.94 輛 |
| **測試集 (Test)** | 2012/09 ~ 2012/12 | 2,609 | 19,319.16 | 138.99 輛 | **0.5824** | 0.5795 | 100.92 輛 |
| **測試集 95% 共形覆蓋**| 2012/09 ~ 2012/12 | 2,609 | — | 區間寬度 526 輛 | **94.40%** 覆蓋率 | $q = \pm 318.81$ 輛 | 誤差僅 0.6% |

---

### 5. OLS 迴歸係數與共線性 (VIF) 診斷

Fitted OLS 迴歸數學方程式：
$$\hat{\text{cnt}} = -6.574 + 70.75 \cdot \text{yr} - 16.52 \cdot \text{holiday} - 25.45 \cdot \text{workingday} + 280.08 \cdot \text{temp} + 65.28 \cdot \text{atemp} + 50.11 \cdot \text{hum} - 32.10 \cdot \text{windspeed} + 38.64 \cdot \text{season\_summer} + 30.79 \cdot \text{season\_fall} + 57.29 \cdot \text{season\_winter} - 8.59 \cdot \text{weather\_mist} - 56.55 \cdot \text{weather\_bad} - 62.86 \cdot \text{hr\_sin} - 55.50 \cdot \text{hr\_cos} - 18.29 \cdot \text{hr\_sin2} - 26.58 \cdot \text{hr\_cos2} + 164.92 \cdot \text{is\_rush\_hour} - 223.35 \cdot \text{temp\_hum}$$

#### 係數與變異數膨脹因子表 (Coefficients & VIF Full Output)
- `temp` (氣溫)：$B = +280.08, \beta = +0.4107, t = 7.942, p < .001$，$\text{VIF} = \mathbf{92.47}$（嚴重共線性）
- `atemp` (體感溫度)：$B = +65.28, \beta = +0.0860, t = 1.835, p = .0666$，$\text{VIF} = \mathbf{75.97}$（嚴重共線性）
- `temp_hum` (溫濕交互)：$B = -223.35, \beta = -0.2490, t = -10.903, p < .001$，$\text{VIF} = \mathbf{18.04}$（嚴重共線性）
- `hum` (濕度)：$B = +50.11, \beta = +0.0736, t = 4.733, p < .001$，$\text{VIF} = 8.37$（中度注意）
- `is_rush_hour` (尖峰指示器)：$B = +164.92, \beta = +0.3906, t = 64.255, p < .001$，$\text{VIF} = 1.28$（良好正常，通勤尖峰平均借車增加 165 輛）
- 其他 13 個特徵：$\text{VIF} < 5$（無共線性疑慮）

#### ANOVA 變異數分解表 (Analysis of Variance)
- **迴歸模型 (Regression)**：$SS = 131,847,583.10 \mid df = 18 \mid MS = 7,324,865.73 \mid F = 1,343.35 \mid p < .001$ ***
- **殘差誤差 (Residual)**：$SS = 56,751,493.52 \mid df = 10,408 \mid MS = 5,452.68$
- **總變異 (Total)**：$SS = 188,599,076.62 \mid df = 10,426$

---

### 6. 環境安裝與執行重現指引 (How to Run)

#### Step 1: 安裝 Python 相依套件
```bash
pip install -r requirements.txt
```

#### Step 2: 執行完整模型管線 (作業主程式)
```bash
python 7115056046_hw2.py
```
*(或執行 `python main.py`，完整執行時間約 15 秒，將產出全部統計報表與圖檔)*

#### Step 3: 啟動本機成果展示網頁
```bash
python web_server.py
```
在瀏覽器開啟：`http://localhost:8080`

---

### 7. 互動成果呈現網頁與設計參考說明 (Interactive Web & GitHub Pages)

本專案實作了高品質之資料科學互動式成果呈現網頁，並已發布至 GitHub Pages：
- **線上展示網址**：[https://chen-fanny.github.io/HW2-Multiple-Linear-regression/](https://chen-fanny.github.io/HW2-Multiple-Linear-regression/)
- **版面設計參考來源**：
  - 參考網站：[StatsUnlock — Multiple Linear Regression Calculator](https://statsunlock.com/multiple-linear-regression-calculator/)
  - **參考版面特色與技術吸納**：
    1. **OLS 模型摘要統計卡片 (Model Summary Cards)**：呈現 $R^2$、Adjusted $R^2$、$F$-statistic、$p$-value、RMSE 及 $n/k$ 樣本比，搭配顯著性判定橫幅。
    2. **彩色標記數學迴歸方程式 (Mathematical Equation Banner)**：動態展示未標準化 OLS 迴歸方程式。
    3. **完整迴歸係數與 VIF 檢驗表**：完整輸出未標準化係數 ($B$)、標準誤 ($SE$)、標準化係數 ($\beta$)、$t$ 統計量、$p$ 值、95% 信賴區間與 VIF 色彩預警標籤。
    4. **ANOVA 變異數分解表**：輸出迴歸、殘差與總變異之平方和、自由度、均方與 F 檢定。
    5. **四大經典統計診斷圖**：預測 vs 實際散佈圖、殘差 vs 擬合圖、標準化係數條形圖、殘差常態分佈鐘形圖。
    6. **模型假設檢驗燈號**：共線性 (VIF)、殘差獨立性 (Durbin-Watson)、殘差常態性 (Skewness) 與樣本規模比。
    7. **學術報告與 APA 7 寫作論述**：提供標準 APA 格式之統計推論論述範例。
    8. **即時情境模擬器 (What-If Interactive Calculator)**：自由調整小時、氣溫、濕度、風速、工作日、季節與天氣，毫秒級即時計算點預測值與 95% 共形預測區間 $[ \max(0, \hat{y} - q), \hat{y} + q ]$。

---

### 8. 繳交作業檢查清單 (Submission Checklist)
- [x] `7115056046_hw2.py`：包含完整資料清洗、五種特徵選取、多元線性迴歸、共形預測與圖表輸出程式碼。
- [x] `requirements.txt`：列出執行所需相依套件。
- [x] `README.md`：提供完整繁體中文專案說明、重現指引與 StatsUnlock 版面參考致謝。
- [x] `results/`：包含 12 張評估圖檔與 6 份 CSV 評估與共線性報表。
- [x] `index.html`、`style.css`、`app.js`：GitHub Pages 互動式成果呈現儀表板與即時預測模擬器。
- [x] `StudentID_hw2.zip`：整合程式碼、互動網頁、模型與成果圖表之標準格式壓縮檔。
- [ ] **學生個人完成項目**：
  - **手寫學術報告 (Handwritten Report)**：依據作業規定親筆手寫完成紙本或掃描報告。
  - 於報告手寫區塊附上 NotebookLM 摘要 (至少 100 字) 與 GPT 提問歷程。
