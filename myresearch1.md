## 研究想法

### 1. 系統核心架構：離散事件模擬 (DES) 與 雙重故障

首先，我們必須定義機器的物理行為。這部分主要參考 **Ghaleb et al. (2020)** 的多狀態模型。

#### **A. 機器狀態模型 (Discrete Multi-state Deterioration)**
將機器狀態 $S_k(t)$ 離散化為 $0, 1, \dots, K$：
*   **State 0**: 全新 (Perfect)。
*   **State 1 ~ K-1**: 衰退中 (Degrading)。加工時間會隨著狀態增加而變長（效率降低）。
*   **State K**: 故障 (Failed due to deterioration)。必須進行大修 (Replacement)。

#### **B. 雙重故障模式 (Dual Failure Modes)**
這是您構想中的亮點，能大幅增加模擬的真實性：
1.  **老化故障 (Deterioration Failure)**：
    *   機制：隨時間推移，狀態從 $d$ 轉移到 $d+1$。當到達 $K$ 時停機。
    *   對策：**預防性維護 (PM)** 或 **狀態基礎維護 (CBM)**。
2.  **隨機故障 (Random Breakdown)**：
    *   機制：在任何運作狀態下（State $0 \sim K-1$），依照 Poisson Process (指數分佈) 隨機發生。
    *   對策：**最小修復 (Minimal Repair)**。修好後機器回到故障前的狀態 $d$，不改善健康度。

---

### 2. 狀態空間設計 (State Space Design)

這是您最關心的部分。在動態且事件驅動的環境下，Agent 需要知道「現在有多急？」以及「機器有多爛？」。

參考 **Lei et al. (2024)** (處理動態工件) 與 **Ghaleb et al. (2020)** (處理多狀態衰退)，建議的狀態向量 $S_t$ 應包含以下 **四大類特徵** (假設有 $M$ 台機器)：

#### **第一類：機器健康特徵 (Machine Health Features)** - *參考 Ghaleb et al.*
Agent 需要這些資訊來判斷是否該維修，從而取代原本的 STW 與 Local Search。
1.  **當前衰退狀態 (Current Degradation State)**: $[d_1, d_2, \dots, d_M]$。正規化為 $d_k / K$。
2.  **累計運作時間 (Time Since Last Transition)**: 機器在當前狀態已經撐了多久？這有助於預測下一次狀態轉移（老化）何時發生。
3.  **故障歷史 (Failure History)**: 過去一段時間內的隨機故障次數（反映機器穩定性）。

#### **第二類：機器運作特徵 (Machine Operational Features)**
1.  **機器狀態 (Machine Status)**: One-hot 編碼或數值編碼 (0: Idle, 1: Busy, 2: Repairing, 3: PM)。
2.  **剩餘加工時間 (Remaining Processing Time)**: 如果機器忙碌，還需要多久才釋放？（若是 Idle 則為 0）。

#### **第三類：工件與隊列特徵 (Job & Queue Features)** - *參考 Lei et al.*
由於工件是動態到達的，我們不能輸入所有工件資訊，只能輸入「統計特徵」。
1.  **隊列長度 (Queue Length)**: 目前有多少工件在排隊？（反映負載壓力）。
2.  **平均緊迫度 (Average Tardiness/Urgency)**: $\frac{1}{N_{queue}} \sum (\text{Current Time} - \text{Due Date})$。數值越大代表越急，Agent 應傾向生產而非維修。
3.  **平均加工時間 (Average Processing Time)**: 隊列中工件的平均工時。

#### **第四類：資源特徵 (Resource Features)** - *參考 An et al.*
1.  **可用維修工數量 (Available Maintenance Crews)**: $Q_{current} / Q_{limit}$。
    *   **關鍵作用**：這能幫助 Agent 學會 **「當沒有維修工時，不要選擇維修動作」**，從而取代 STW 的硬性規則。

---

### 3. 動作空間與執行邏輯 (Action & Execution)

您提到「Agent 只能看到目前的工作」，這非常符合 **Lei et al. (2024)** 的 **"Job Buffer"** 概念。

*   **觸發時機 (Trigger)**：當「有機器閒置 (Machine Idle)」且「Buffer 中有工件」時。
*   **動作定義 (Action)**：$A_t \in \{ \text{Dispatch Rules} \} \times \{ \text{Maintenance Actions} \}$
    *   **派工規則**: SPT, EDD, FIFO, LPT... (用來從 Buffer 選一個工件給這台閒置機器)。
    *   **維護決策**:
        *   **None (Z)**: 不修，直接做工件。
        *   **PM (P)**: 先做預防性維護 (改善狀態)，再做工件。
        *   *(註：隨機故障的修復是強制性的，不需要 Agent 決定)*

---

### 4. 獎勵函數設計 (Reward Function)

目標是減少 **Total Tardiness**。

*   **Step Reward**: $r_t = - \sum_{j \in \text{System}} \max(0, \text{Current Time} - \text{Due Date}_j)$
    *   這是 **累積延遲懲罰**。只要工件在系統中且已過期，每一秒都會扣分。這會逼迫 Agent 盡快把快過期的工件做完。
*   **維護懲罰**: 若選擇 PM，額外扣除固定成本（避免 Agent 沒事一直修）。
*   **無效動作懲罰**: 若在 $Q_{current}=0$ 時選擇 PM，給予大額負獎勵並強制轉為「不維修」。

---

### 5. 實作建議 (Implementation Roadmap)

基於您現有的程式碼，建議的修改路徑如下：

1.  **修改 `Machine` 類別**：
    *   加入 `state` (0~K)。
    *   實作 `transition_matrix` (老化機率)。
    *   加入 `random_breakdown` 邏輯 (Poisson)。
2.  **建立 `JobGenerator`**：
    *   使用 `simpy` 或手寫 `next_arrival_time` 來生成動態工件。
3.  **重寫 `DFJSP_Env` 為事件驅動**：
    *   使用 `heapq` 管理 [Job Arrival, Op Finish, Breakdown, Repair Finish] 四種事件。
    *   **狀態回傳**：改為上述的 4 類特徵向量。
4.  **移除 STW 與 LS**：
    *   將資源限制邏輯移入 `step` 函數的 Reward 懲罰機制中。
    *   完全移除 `LocalSearch` 類別。

這個架構將會是一個非常標準且強大的 **Sim-to-Real** 排程模型，完全具備發表高品質期刊的潛力。