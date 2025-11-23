import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import matplotlib.transforms as mtransforms

# --- 1. 解析パラメータ ---
FPS_VIDEO = 30.0
SENSOR_SAMPLE_MS = 10.0
DATA_START_ROW = 62
PIXELS_PER_MM = 4.6402

# ★★★ ゼロ傾きと見なすための最大傾き閾値 (固定) ★★★
RAW_SLOPE_THRESHOLD_NPS = 2 

# --- 2. Matplotlibスタイル設定 ---
plt.rcParams.update({
    'font.size': 24, # 基本フォント
    'axes.labelsize': 36,  # 軸タイトルサイズ (大きく)
    'xtick.labelsize': 24, # 目盛りサイズ (大きく)
    'ytick.labelsize': 24, 
    'legend.fontsize': 24, 
    'figure.figsize': (15, 10), 
    'font.family': 'Arial'
})

def get_task_name(index):
    """ ファイルのインデックスに基づき、タスク名を返す (例: Task 1-1) """
    if index < 3:
        return 'Task 1-1 (1st-2nd)'
    elif index < 6:
        return 'Task 1-2 (2nd-3rd)'
    elif index < 9:
        return 'Task 1-3 (3rd-4th)'
    elif index < 12: 
        return 'Task 1-4 (4th-5th)'
    else:
        return f'Task_Unknown_{index+1}'


# --- 3. コア関数: ゼロ傾き付近の把持力の抽出 ---
def get_zero_slope_forces(force_series, sensor_period_s):
    """
    力の傾き dF/dt が RAW_SLOPE_THRESHOLD_NPS 未満の区間の把持力を抽出する。
    """
    
    # 1. 傾き (dF/dt) を計算 (生データを使用)
    time_steps = force_series.index.to_series().diff().dt.total_seconds().fillna(sensor_period_s)
    # Series.fillna(method) は非推奨なので、Series.bfill().diff()を使用
    force_derivative_raw = force_series.bfill().diff().fillna(0) / time_steps 
    
    # 2. ゼロ傾きフィルター: 傾きの絶対値が閾値未満の区間
    zero_slope_filter = np.abs(force_derivative_raw) < RAW_SLOPE_THRESHOLD_NPS
    
    # 3. 安定していると見なされる区間の力データのみを抽出
    force_series_filtered = force_series.loc[zero_slope_filter.index]
    zero_slope_forces = force_series_filtered[zero_slope_filter].values
    
    print(f"  [INFO] 抽出データ点数: {len(zero_slope_forces)} 件 (閾値: {RAW_SLOPE_THRESHOLD_NPS} N/s)")
    
    return zero_slope_forces


# --- 4. コア関数: ヒストグラムの描画 ---
def plot_turning_point_histogram(all_turning_point_forces, output_dir):
    """
    全試行の安定把持力データを統合し、2N以上の範囲のみでヒストグラムを描画する。
    """
    combined_forces = np.concatenate(all_turning_point_forces)
     
    # 2Nより上のデータのみをフィルタリング (F > 1.0 N)
    filtered_forces = combined_forces[combined_forces > 0.8]
    
    if len(filtered_forces) == 0:
        print("[警告] 1Nより大きいデータがありませんでした。描画をスキップします。")
        return
        
    plt.figure(figsize=(10, 8))
    
    # ヒストグラムの描画
    sns.histplot(filtered_forces, bins=30, kde=True, 
                 stat='count', color='darkblue', edgecolor='none')

    # 軸タイトルとラベルの調整
    plt.xlabel(f"Force(N)", fontsize=36) 
    plt.ylabel("Count", fontsize=36) 
    
    # 線や凡例は削除済み
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "summary_ZeroSlope_Force_Histogram_F_gt_2N_final.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"\n✓ ゼロ傾き把持力ヒストグラムを保存しました: {os.path.basename(plot_path)}")


# --- 5. 処理関数 (データ読み込み) ---
def aggregate_trial_data(force_path, a, b, col_c, col_d):
    """
    v37.2: ゼロ傾き時の把持力マグニチュードを抽出する。
    """
    try:
        # --- ステップ1: 把持力読み込み ---
        df_force = pd.read_csv(force_path, skiprows=DATA_START_ROW, header=None, encoding='cp932')
        strain_C = pd.to_numeric(df_force[col_c], errors='coerce')
        strain_D = pd.to_numeric(df_force[col_d], errors='coerce')
        if strain_C.isnull().all() or strain_D.isnull().all():
             return None
             
        strain_mean = (strain_C + strain_D) / 2
        force_mN = (a * strain_mean**2) + (b * strain_mean)
        force_N = force_mN / 1000.0
        
        # 元のオフセット処理 (開始点オフセット)
        force_N_offsetted = force_N - force_N.iloc[0]
        
        sensor_period_s = SENSOR_SAMPLE_MS / 1000.0
        time_s_force = df_force.index * sensor_period_s
        force_series = pd.Series(force_N_offsetted.values, index=pd.to_timedelta(time_s_force, unit='s'))

        # --- ステップ2: ゼロ傾き時の把持力の抽出 ---
        zero_slope_force_array = get_zero_slope_forces(
            force_series, 
            sensor_period_s
        )
        return zero_slope_force_array

    except Exception as e:
        print(f"  [エラー] ファイル {os.path.basename(force_path)} 処理中に問題が発生しました: {e}")
        return None


# --- 6. メイン実行ブロック ---
if __name__ == "__main__":
    
    print("--- v37.2: STEP 1: ゼロ傾き把持力ヒストグラム可視化 (最終版) ---")
    print("===============================================================")

    # --- 1. ハードコードされた定数 ---
    force_dir = r"C:\Users\haruk\OneDrive\ドキュメント\修士\実験データ\20251022\解析結果\カテーテル\時間同期_把持力_yuki_1"
    tip_dir = r"C:\Users\haruk\OneDrive\ドキュメント\修士\実験データ\20251022\解析結果\カテーテル\先端検出結果_v13.6_Debug\yuki_1"
    output_dir = r"C:\Users\haruk\OneDrive\ドキュメント\修士\実験データ\20251022\解析結果\カテーテル\総合解析_yuki_1"

    a_coeff = 0.019
    b_coeff = 5.647
    col_c = 4
    col_d = 5
    
    if not os.path.isdir(force_dir):
        print(f"[エラー] 把持力フォルダが見つかりません: {force_dir}")
        exit()
    os.makedirs(output_dir, exist_ok=True)

    # --- 2. ファイルの紐付け ---
    force_files = sorted(glob.glob(os.path.join(force_dir, "*.csv"))) 
    
    if not force_files:
        print(f"[エラー] 把持力CSVファイルが見つかりません。")
        exit()
        
    print(f"\n--- 3. 全 {len(force_files)} 件のファイルからゼロ傾き把持力データを収集します ---")
    print(f"   [安定判定基準]: {RAW_SLOPE_THRESHOLD_NPS} N/s")
    print(f"   [出力フィルタ]: F > 1.0 N")

    # --- 4. ゼロ傾き把持力データの収集と計算 ---
    all_zero_slope_forces = []
    
    for force_path in force_files:
        
        zero_slope_force_array = aggregate_trial_data(
            force_path, 
            a_coeff, b_coeff, col_c, col_d
        )
        if zero_slope_force_array is not None:
            all_zero_slope_forces.append(zero_slope_force_array)

    if not all_zero_slope_forces:
        print("\n[!!! エラー !!!] ゼロ傾き把持力データを取得できませんでした。処理を終了します。")
        exit()

    # --- 5. ヒストグラムの描画 ---
    plot_turning_point_histogram(all_zero_slope_forces, output_dir)