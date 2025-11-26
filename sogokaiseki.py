import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.transforms as mtransforms

# --- 1. 解析パラメータ ---
FPS_VIDEO = 30.0
SENSOR_SAMPLE_MS = 10.0
DATA_START_ROW = 62
PIXELS_PER_MM = 4.6402

# ★★★ Matplotlibスタイル設定 ★★★
plt.rcParams.update({
    'font.size': 24,           # 基本フォント
    'axes.labelsize': 36,      # 軸タイトルサイズ (大きく)
    'xtick.labelsize': 28,     # X軸目盛りサイズ (大きく)
    'ytick.labelsize': 28,     # Y軸目盛りサイズ (大きく)
    'legend.fontsize': 24,     # 凡例サイズ (大きく)
    'figure.figsize': (12, 18), # グラフサイズを標準的なプレゼン用に再調整
    'font.family': 'Arial'
})

def get_task_name(index):
    """
    ファイルのインデックス(0から)に基づき、タスク名を返す。
    (4条件x3試行の仮定)
    """
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

# ★★★ ラベル簡略化ヘルパー関数 ★★★
def simplify_label(label_text):
    """ 'Task 1-1 (1st-2nd)' -> '1st-2nd' に簡略化 """
    match = re.search(r'\((.*?)\)', label_text)
    # 括弧内のテキストを抽出し、ハイフンの後に改行を入れて縦に表示
    return match.group(1).replace('-', '-\n') if match else label_text

# --- 3. コア関数: プッシュイベントの検出ロジック ---
def find_push_events(df_merged, force_threshold, min_duration_s, min_gap_s):
    """ 力の絶対値で閾値処理を行い、イベントを検出・フィルタリングする。"""
    
    df_merged['is_pushing'] = df_merged['force'] > force_threshold
    state_changes = df_merged['is_pushing'].astype(int).diff()
    start_times = df_merged[state_changes == 1].index
    end_times = df_merged[state_changes == -1].index
    
    initial_events = []
    if not df_merged.empty and df_merged['is_pushing'].iloc[0]:
        if not end_times.empty:
            initial_events.append((df_merged.index[0], end_times[0]))
            end_times = end_times[1:]
        else:
            initial_events.append((df_merged.index[0], df_merged.index[-1]))
            
    for start in start_times:
        matching_end = end_times[end_times > start]
        if not matching_end.empty:
            initial_events.append((start, matching_end[0]))

    if not initial_events:
        return []

    # フィルタリングと結合ロジック (省略なし)
    filtered_events = []
    for start, end in initial_events:
        if (end - start).total_seconds() >= min_duration_s:
            filtered_events.append((start, end))

    if not filtered_events: return []

    merged_events = []
    current_event_start, current_event_end = filtered_events[0]
    for next_start, next_end in filtered_events[1:]:
        if (next_start - current_event_end).total_seconds() <= min_gap_s:
            current_event_end = next_end
        else:
            merged_events.append((current_event_start, current_event_end))
            current_event_start, current_event_end = next_start, next_end
            
    merged_events.append((current_event_start, current_event_end))
    
    return merged_events


# --- 4. コア関数: データ集計とイベント検出 ---
def aggregate_trial_data(force_path, tip_path, a, b, col_c, col_d, fps, task_name, px_per_mm, 
                         force_threshold, force_smoothing_window_ms, min_push_duration_s, min_gap_duration_s):
    """
    把持力と先端座標を処理し、プッシュイベントを検出して集計する。
    """
    base_name = os.path.splitext(os.path.basename(tip_path))[0].replace("_tips_coordinates", "")
    print(f"   -> {task_name}: {base_name}")

    try:
        # --- ステップ1: 把持力読み込み & 平滑化 ---
        df_force = pd.read_csv(force_path, skiprows=DATA_START_ROW, header=None, encoding='cp932')
        strain_C = pd.to_numeric(df_force[col_c], errors='coerce')
        strain_D = pd.to_numeric(df_force[col_d], errors='coerce')
        if strain_C.isnull().all() or strain_D.isnull().all():
            print(f"       [エラー] ひずみ列が空です。スキップします。")
            return None, None
            
        strain_mean = (strain_C + strain_D) / 2
        force_mN = (a * strain_mean**2) + (b * strain_mean)
        force_N = force_mN / 1000.0
        force_N_offsetted = force_N - force_N.iloc[0] # 開始点オフセット
        
        sensor_period_s = SENSOR_SAMPLE_MS / 1000.0
        time_s_force = df_force.index * sensor_period_s
        force_series = pd.Series(force_N_offsetted.values, index=pd.to_timedelta(time_s_force, unit='s'))

        window_size = int(force_smoothing_window_ms / SENSOR_SAMPLE_MS)
        if window_size < 1: window_size = 1
        force_series_smooth = force_series.rolling(window=window_size, min_periods=1, center=True).mean()


        # --- ステップ2: 座標読み込み & マージ ---
        df_tip = pd.read_csv(tip_path)
        if df_tip.empty or 'smooth_cath_x' not in df_tip.columns:
            print(f"       [エラー] 先端座標CSVが空です。スキップします。")
            return None, None
        df_tip['time_s'] = df_tip['frame'] / fps
        df_tip.set_index(pd.to_timedelta(df_tip['time_s'], unit='s'), inplace=True)

        force_aligned = force_series_smooth.reindex(df_tip.index, method='nearest', tolerance=pd.Timedelta(sensor_period_s, unit='s'))
        df_merged = df_tip.copy()
        df_merged['force'] = force_aligned 
        df_merged.dropna(subset=['smooth_cath_x', 'smooth_cath_y', 'force'], inplace=True)
        if df_merged.empty:
            print(f"       [エラー] マージに失敗しました。スキップします。")
            return None, None

        # --- ステップ3: プッシュイベントの検出 ---
        push_events_indices = find_push_events(
            df_merged, force_threshold, min_push_duration_s, min_gap_duration_s
        )
        trial_push_count = len(push_events_indices)
        
        # 始点からの直線距離 (Effective Distance) を計算
        coords = df_merged[['smooth_cath_x', 'smooth_cath_y']].values
        start_coord = coords[0]
        effective_distance_px = np.sqrt(((coords - start_coord)**2).sum(axis=1))
        df_merged['effective_distance_mm'] = effective_distance_px / px_per_mm

        push_event_details = [] 

        if trial_push_count == 0:
            print(f"       [警告] プッシュイベントが0回でした (閾値: {force_threshold} N, フィルタ適用後)。")
        
        for i, (start_time, end_time) in enumerate(push_events_indices):
            
            try:
                event_df = df_merged.loc[start_time:end_time]
            except Exception:
                event_df = df_merged[start_time:end_time]
                
            if len(event_df) < 2: continue 

            duration_s = (event_df.index[-1] - event_df.index[0]).total_seconds()
            
            start_dist_mm = event_df['effective_distance_mm'].iloc[0]
            end_dist_mm = event_df['effective_distance_mm'].iloc[-1]
            distance_mm = max(0, end_dist_mm - start_dist_mm)

            max_force_N = event_df['force'].max() 

            push_event_details.append({
                'task_name': task_name, 'trial_name': base_name, 'push_number': i + 1, 
                'duration_s': duration_s, 'distance_mm': distance_mm, 'max_force_N': max_force_N,
            })
            
        trial_summary = {
            'task_name': task_name, 'trial_name': base_name, 'total_push_count': trial_push_count
        }
        
        return trial_summary, push_event_details

    except Exception as e:
        print(f"     [!!! 重大なエラー !!!] {base_name} の処理中にエラーが発生しました: {e}")
        return None, None


# --- 5. メイン実行ブロック ---
if __name__ == "__main__":
    
    print("--- v32.2: Task 1 プッシュ解析 (グラフ最終調整) ---")
    print("解析を開始します。")
    print("=====================================================")

    # --- 1. ハードコードされた定数 ---
    # ★★★ フォルダパス ★★★
    force_dir = r"C:\Users\haruk\OneDrive\ドキュメント\修士\実験データ\20251022\解析結果\カテーテル\時間同期_把持力_yuki_1"
    tip_dir = r"C:\Users\haruk\OneDrive\ドキュメント\修士\実験データ\20251022\解析結果\カテーテル\先端検出結果_v13.6_Debug\yuki_1"
    output_dir = r"C:\Users\haruk\OneDrive\ドキュメント\修士\実験データ\20251022\解析結果\カテーテル\総合解析_yuki_1"

    # ★★★ キャリブレーション係数 ★★★
    a_coeff = 0.019
    b_coeff = 5.647
    col_c = 4
    col_d = 5
    
    if not os.path.isdir(force_dir):
        print(f"[エラー] 把持力フォルダが見つかりません: {force_dir}")
        exit()
    if not os.path.isdir(tip_dir):
        print(f"[エラー] 先端座標フォルダが見つかりません: {tip_dir}")
        exit()
    os.makedirs(output_dir, exist_ok=True)

    # --- 2. ユーザー入力: プッシュ判定のフィルタ ---
    print("\n--- 4. プッシュ判定のフィルタを最終確認してください ---")
    try:
        push_threshold = float(input("   -> 1. プッシュ判定閾値 (N) (例: 4.0): "))
        print(" (推奨初期値: 100, 0.15, 0.2)")
        smoothing_ms = float(input("   -> 2. 移動平均の窓幅 (ms) (例: 100): "))
        min_duration = float(input("   -> 3. 最小プッシュ時間 (s) (例: 0.15): "))
        min_gap = float(input("   -> 4. 最小ギャップ時間 (s) (例: 0.2): "))
            
    except ValueError as e:
        print(f"[エラー] 無効な入力です: {e}。処理を終了します。")
        exit()
 
    # --- 3. ファイルの紐付け ---
    force_files = sorted(glob.glob(os.path.join(force_dir, "*.csv"))) 
    tip_files = sorted(glob.glob(os.path.join(tip_dir, "*_tips_coordinates.csv"))) 
    
    if not force_files or not tip_files:
        print(f"[エラー] 把持力または先端座標フォルダにCSVが見つかりません。")
        exit()
        
    if len(force_files) != len(tip_files):
        print(f"[!!! 警告 !!!] ファイル数が一致しません！ ({len(force_files)} vs {len(tip_files)})")
        print("   (ただし、続行します...)")

    file_map = dict(zip(force_files, tip_files))
        
    print(f"\n--- 5. {len(file_map)} 件のファイルペアで解析を実行します ---")

    # --- 4. 解析の実行 ---
    all_trial_summaries = []
    all_push_events = []
    
    for i, (force_path, tip_path) in enumerate(file_map.items()):
        
        task_name = get_task_name(i)
        
        trial_summary, push_event_details = aggregate_trial_data(
            force_path, tip_path, 
            a_coeff, b_coeff, col_c, col_d,
            FPS_VIDEO, task_name,
            PIXELS_PER_MM,
            push_threshold,              
            smoothing_ms,              
            min_duration,              
            min_gap                     
        )
        if trial_summary: all_trial_summaries.append(trial_summary)
        if push_event_details: all_push_events.extend(push_event_details)

    if not all_trial_summaries:
        print("\n[!!! エラー !!!] 解析できるデータがありませんでした。")
        exit()

    # --- 5. 集計結果をCSVとグラフに出力 ---
    print("\n" + "="*50)
    print("--- 全ての解析が完了しました。集計結果を出力します ---")
    
    df_trials = pd.DataFrame(all_trial_summaries)
    df_pushes = pd.DataFrame(all_push_events)
    
    if df_pushes.empty:
        print("[警告] フィルタ適用後、有効なプッシュイベントが1件も見つかりませんでした。")
        df_pushes = pd.DataFrame(columns=['task_name', 'distance_mm', 'duration_s']) 

    
    csv_path_trials = os.path.join(output_dir, f"trial_summary_pushes.csv")
    csv_path_events = os.path.join(output_dir, f"push_events_summary.csv")
    df_trials.to_csv(csv_path_trials, index=False)
    df_pushes.to_csv(csv_path_events, index=False, encoding='utf-8-sig')
    print(f"\n✓ 試行ごとのプッシュ回数CSVを保存しました: {os.path.basename(csv_path_trials)}")
    print(f"✓ 全プッシュイベント詳細CSVを保存しました: {os.path.basename(csv_path_events)}")

    task_order = sorted(df_trials['task_name'].unique())
    
    # グラフ1: プッシュ回数 (試行ごとの合計)
    try:
        fig1, ax1 = plt.subplots(figsize=(12, 18))
        sns.barplot(x='task_name', y='total_push_count', data=df_trials, 
                     capsize=0.1, errorbar='sd', order=task_order, ax=ax1) 
        
        ax1.set_xticklabels([simplify_label(label.get_text()) for label in ax1.get_xticklabels()])
        ax1.set_xlabel("Curve Position", fontsize=36) 
        ax1.set_ylabel("Push Count (per Trial)", fontsize=36)
        ax1.tick_params(axis='both', which='major', labelsize=28)
        
        ax1.grid(True, linestyle='--', alpha=0.5, axis='y')
        fig1.tight_layout()
        plot_path_force = os.path.join(output_dir, "summary_graph_PushCount.png")
        fig1.savefig(plot_path_force)
        plt.close(fig1)
        print(f"✓ プッシュ回数のグラフを保存しました: {os.path.basename(plot_path_force)}")
    except Exception as e:
        print(f"   [エラー] プッシュ回数グラフの作成に失敗: {e}")

    # グラフ2: プッシュ1回あたりの「平均」前進距離
    try:
        fig2, ax2 = plt.subplots(figsize=(12, 18))
        sns.barplot(x='task_name', y='distance_mm', data=df_pushes, 
                     capsize=0.1, errorbar='sd', order=task_order, ax=ax2)
        
        ax2.set_xticklabels([simplify_label(label.get_text()) for label in ax2.get_xticklabels()])
        ax2.set_xlabel("Curve Position", fontsize=36)
        ax2.set_ylabel("Average Distance per Push (mm)", fontsize=36)
        
        ax2.tick_params(axis='both', which='major', labelsize=28)
        
        ax2.grid(True, linestyle='--', alpha=0.5, axis='y')
        fig2.tight_layout()
        plot_path_work = os.path.join(output_dir, "summary_graph_AvgDistPerPush.png")
        fig2.savefig(plot_path_work)
        plt.close(fig2)
        print(f"✓ 平均前進距離/プッシュのグラフを保存しました: {os.path.basename(plot_path_work)}")
    except Exception as e:
        print(f"   [エラー] 平均前進距離グラフの作成に失敗: {e}")
        
    # グラフ3: プッシュ1回あたりの「平均」時間
    try:
        fig3, ax3 = plt.subplots(figsize=(12, 18))
        sns.barplot(x='task_name', y='duration_s', data=df_pushes, 
                     capsize=0.1, errorbar='sd', order=task_order, ax=ax3)
        
        ax3.set_xticklabels([simplify_label(label.get_text()) for label in ax3.get_xticklabels()])
        ax3.set_xlabel("Curve Position", fontsize=36)
        ax3.set_ylabel("Average Duration per Push (s)", fontsize=36)
        
        ax3.tick_params(axis='both', which='major', labelsize=28)
        
        ax3.grid(True, linestyle='--', alpha=0.5, axis='y')
        fig3.tight_layout()
        plot_path_work = os.path.join(output_dir, "summary_graph_AvgDurationPerPush.png")
        fig3.savefig(plot_path_work)
        plt.close(fig3)
        print(f"✓ 平均時間/プッシュのグラフを保存しました: {os.path.basename(plot_path_work)}")
    except Exception as e:
        print(f"   [エラー] 平均時間グラフの作成に失敗: {e}")

    print("\n" + "="*50)
    print(f"--- {os.path.basename(output_dir)} の集計が完了しました ---")