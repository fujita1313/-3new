import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms


# --- 1. 解析パラメータ ---
FPS_VIDEO = 30.0                
SENSOR_SAMPLE_MS = 10.0         
DATA_START_ROW = 62             
VELOCITY_THRESHOLD_PPS = 35.0   
# ★★★ 追加: ピクセル/mm 換算比 ★★★
PIXELS_PER_MM = 4.6402

plt.rcParams.update({
    'font.size': 16, 
    'axes.labelsize': 22, 
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.figsize': (15, 10), 
    'font.family': 'Arial'
})

# --- ★★★ 修正: analyze_single_file (v28.4) ★★★ ---
def analyze_single_file(force_path, tip_path, output_dir, a, b, col_c, col_d, fps, vel_thresh_per_frame, px_per_mm):
    """
    v28.4: 距離の指標を「mm」単位に換算してグラフ化
    """
    
    base_name = os.path.splitext(os.path.basename(tip_path))[0].replace("_tips_coordinates", "")
    print(f"\n--- 処理中: {base_name} ---")

    try:
        # --- ステップ1 & 2: ファイル読み込み (v28.3 と同じ) ---
        df_force = pd.read_csv(force_path, skiprows=DATA_START_ROW, header=None, encoding='cp932')
        strain_C = pd.to_numeric(df_force[col_c], errors='coerce')
        strain_D = pd.to_numeric(df_force[col_d], errors='coerce')
        if strain_C.isnull().all() or strain_D.isnull().all():
             print(f"  -> [エラー] ひずみ列 {col_c} または {col_d} が空です。スキップします。")
             return
        strain_mean = (strain_C + strain_D) / 2
        x = strain_mean - strain_mean.iloc[0] 
        force_N = (a * x**2) + (b * x) 
        sensor_period_s = SENSOR_SAMPLE_MS / 1000.0
        time_s_force = df_force.index * sensor_period_s
        force_series = pd.Series(force_N.values, index=pd.to_timedelta(time_s_force, unit='s'))

        df_tip = pd.read_csv(tip_path)
        if df_tip.empty or 'smooth_cath_x' not in df_tip.columns:
            print("  -> [エラー] 先端座標CSVが空か、'smooth_cath_x' 列がありません。スキップします。")
            return
        df_tip['time_s'] = df_tip['frame'] / fps
        df_tip.set_index(pd.to_timedelta(df_tip['time_s'], unit='s'), inplace=True)

        # --- ステップ3: マージ (v28.3 と同じ) ---
        force_aligned = force_series.reindex(df_tip.index, method='nearest', tolerance=pd.Timedelta(sensor_period_s, unit='s'))
        df_merged = df_tip.copy()
        df_merged['force'] = force_aligned
        df_merged.dropna(subset=['smooth_cath_x', 'smooth_cath_y', 'force'], inplace=True)
        if df_merged.empty:
            print("  -> [エラー] 把持力と座標のマージに失敗しました。スキップします。")
            return

        # --- ステップ4: ★★★ 距離(mm)と動的/静的状態を計算 (v28.4 修正) ★★★ ---
        coords = df_merged[['smooth_cath_x', 'smooth_cath_y']].values
        
        # 1. 始点からの直線距離 (ピクセル)
        start_coord = coords[0]
        effective_distance_px = np.sqrt(((coords - start_coord)**2).sum(axis=1))
        # ★★★ mm 単位に換算 ★★★
        df_merged['effective_distance_mm'] = effective_distance_px / px_per_mm

        # 2. 静的/動的 判定 (v28.3 と同じ - ピクセル単位で判定)
        diffs = np.diff(coords, axis=0)
        step_distances_px = np.sqrt((diffs**2).sum(axis=1))
        df_merged['velocity_px_f'] = np.insert(step_distances_px, 0, 0)
        df_merged['is_static'] = df_merged['velocity_px_f'] <= vel_thresh_per_frame
        
        # --- ステップ5: グラフを描画 (v28.4 修正) ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        time_axis = df_merged['time_s']
        
        # --- 上段 (距離グラフ) ---
        ax1.plot(time_axis, df_merged['effective_distance_mm'], color='green', label='Effective Distance')
        # ★★★ 修正: 単位を mm に変更 ★★★
        ax1.set_ylabel('Distance (mm)')
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        y_min, y_max = ax1.get_ylim()

        # ★★★ 修正: ラベルの閾値を mm/s に換算 ★★★
        thresh_mm_s = (vel_thresh_per_frame * fps) / px_per_mm
        ax1.fill_between(time_axis, y_min, y_max, 
                         where=df_merged['is_static'], 
                         facecolor='grey', alpha=0.2, 
                         label=f'Static State (v < {thresh_mm_s:.1f} mm/s)')
        
        ax1.set_ylim(y_min, y_max) 
        ax1.legend(loc='upper left')

        # --- 下段 (把持力グラフ) (v28.3 と同じ) ---
        ax2.plot(time_axis, df_merged['force'], color='red', label='Gripping Force')
        ax2.set_ylabel('Gripping Force (mN)')
        ax2.set_xlabel('Time (s)')
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        y_min, y_max = ax2.get_ylim()
        ax2.set_ylim(0, max(y_max * 1.1, 1.0)) # 最小値を0に (最小値が1未満にならないように)
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        # グラフを保存
        output_path = os.path.join(output_dir, f"{base_name}_force_EFFECTIVE_distance_mm_graph.png") # ファイル名変更
        plt.savefig(output_path)
        plt.close(fig)
        
        print(f"  -> グラフを保存しました: {os.path.basename(output_path)}")
        
    except Exception as e:
        print(f"  [!!! 重大なエラー !!!] {base_name} の処理中にエラーが発生しました: {e}")

# --- メイン実行ブロック (v28.3 とほぼ同じ) ---
if __name__ == "__main__":
    
    print("--- v28.4: Task 1 (S-Curve) 統合解析 (mm単位) ---")
    print("被験者1名分の解析を開始します。")
    print("="*50)

    # --- 1. ユーザー入力: フォルダパス ---
    force_dir = input(r"1. 把持力CSVフォルダのパスを入力 (例: ...\時間同期_把持力_yuki_1): ").strip().replace('"', '')
    tip_dir = input(r"2. 先端座標CSVフォルダのパスを入力 (例: ...\先端検出結果_v13.6_Debug\yuki_1): ").strip().replace('"', '')
    output_dir = input(r"3. グラフ出力先フォルダのパスを入力 (例: ...\統合解析_yuki_1): ").strip().replace('"', '')

    if not os.path.isdir(force_dir):
        print(f"[エラー] 把持力フォルダが見つかりません: {force_dir}")
        exit()
    if not os.path.isdir(tip_dir):
        print(f"[エラー] 先端座標フォルダが見つかりません: {tip_dir}")
        exit()
    os.makedirs(output_dir, exist_ok=True)

    # --- 2. ユーザー入力: キャリブレーション係数 ---
    print("\n--- 4. キャリブレーション係数を入力してください ---")
    try:
        a_coeff = float(input("  -> 係数 'a' (x^2): "))
        b_coeff = float(input("  -> 係数 'b' (x): "))
        col_c = int(input("  -> ひずみ列Cの番号 (0から, 例: 2 or 4): "))
        col_d = int(input("  -> ひずみ列Dの番号 (0から, 例: 3 or 5): "))
    except ValueError:
        print("[エラー] 無効な数値です。処理を終了します。")
        exit()

    # --- 3. ファイルの紐付け ---
    force_files = sorted(glob.glob(os.path.join(force_dir, "*_trimmed.csv")))
    tip_files = sorted(glob.glob(os.path.join(tip_dir, "*_tips_coordinates.csv")))

    if not force_files:
        print(f"[エラー] 把持力フォルダに '*_trimmed.csv' が見つかりません。")
        exit()
    if not tip_files:
        print(f"[エラー] 先端座標フォルダに '*_tips_coordinates.csv' が見つかりません。")
        exit()
        
    if len(force_files) != len(tip_files):
        print(f"[!!! 警告 !!!] ファイル数が一致しません！")
        print(f"  把持力ファイル: {len(force_files)} 件")
        print(f"  先端座標ファイル: {len(tip_files)} 件")
        print("  処理を続行しますが、紐付けがズレる可能性があります。")

    def get_base_name(path, suffix):
        return os.path.basename(path).replace(suffix, "")

    print("\n--- 5. ファイル紐付けの確認 ---")
    print(f"'{os.path.basename(force_dir)}' と '{os.path.basename(tip_dir)}' のファイルを順序で紐付けます。")
    
    file_map = dict(zip(force_files, tip_files))
    for i, (f_path, t_path) in enumerate(file_map.items()):
        f_base = get_base_name(f_path, "_trimmed.csv")
        t_base = get_base_name(t_path, "_tips_coordinates.csv")
        print(f"  [{i+1:02d}] {f_base}  <->  {t_base}")

    print("-" * 50)
    choice = input("この紐付けで解析を実行しますか？ (Yes: y / No: n) [Y/n]: ").strip().lower()
    
    if choice == 'n':
        print("処理を中断しました。")
        exit()

    # --- 6. 解析の実行 ---
    print("\n" + "="*50)
    print(f"--- 解析を実行します（全 {len(file_map)} 件） ---")
    
    # [cite_start]閾値をピクセル/フレームに換算 (v28.3 [cite: 207] と同じ)
    vel_thresh_per_frame = VELOCITY_THRESHOLD_PPS / FPS_VIDEO

    for force_path, tip_path in file_map.items():
        analyze_single_file(
            force_path, tip_path, output_dir,
            a_coeff, b_coeff, col_c, col_d,
            FPS_VIDEO, vel_thresh_per_frame,
            PIXELS_PER_MM # ★★★ 新しい引数を渡す
        )

    print("\n" + "="*50)
    print(f"--- {os.path.basename(output_dir)} の解析が完了しました ---")