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
PIXELS_PER_MM = 4.6402

# --- 2. センサー列インデックスの修正 ---
STRAIN_C_COL_WIRE = 2
STRAIN_D_COL_WIRE = 3 
# ★★★ 修正箇所: ガイドワイヤの先端座標列名 ★★★
TIP_X_COL_NAME = 'smooth_gw_x' 
TIP_Y_COL_NAME = 'smooth_gw_y' 

# --- 3. Matplotlibスタイル設定 ---
plt.rcParams.update({
    'font.size': 20, 
    'axes.labelsize': 36,  # 軸タイトルサイズ (大きく)
    'xtick.labelsize': 24, # X軸目盛りサイズ (大きく)
    'ytick.labelsize': 24, # Y軸目盛りサイズ (大きく)
    'legend.fontsize': 24, # 凡例サイズ (大きく)
    'figure.figsize': (15, 10), 
    'font.family': 'Arial'
})


# --- ★★★ 関数: analyze_guidewire_file (v29.1) ★★★ ---
def analyze_guidewire_file(force_path, tip_path, output_dir, fps, px_per_mm):
    """
    v29.1: ガイドワイヤ解析用。把持力計算なし、生ひずみ平均値をプロット。
    """
    
    base_name = os.path.splitext(os.path.basename(tip_path))[0].replace("_tips_coordinates", "")
    print(f"\n--- 処理中 (ガイドワイヤ): {base_name} ---")

    try:
        # --- ステップ1: センサーデータ読み込み (生データを使用) ---
        df_force = pd.read_csv(force_path, skiprows=DATA_START_ROW, header=None, encoding='cp932')
        strain_C = pd.to_numeric(df_force[STRAIN_C_COL_WIRE], errors='coerce')
        strain_D = pd.to_numeric(df_force[STRAIN_D_COL_WIRE], errors='coerce')
        if strain_C.isnull().all() or strain_D.isnull().all():
             print(f"  -> [エラー] ひずみ列 {STRAIN_C_COL_WIRE} または {STRAIN_D_COL_WIRE} が空です。スキップします。")
             return
        
        raw_strain_mean = (strain_C + strain_D) / 2
        
        sensor_period_s = SENSOR_SAMPLE_MS / 1000.0
        time_s_force = df_force.index * sensor_period_s
        raw_strain_series = pd.Series(raw_strain_mean.values, index=pd.to_timedelta(time_s_force, unit='s'))

        # --- ステップ2: 先端座標読み込み (ガイドワイヤ列名を使用) ---
        df_tip = pd.read_csv(tip_path)
        if df_tip.empty or TIP_X_COL_NAME not in df_tip.columns:
            print(f"  -> [エラー] 先端座標CSVが空か、'{TIP_X_COL_NAME}' 列がありません。スキップします。")
            return
        
        df_tip['time_s'] = df_tip['frame'] / fps
        df_tip.set_index(pd.to_timedelta(df_tip['time_s'], unit='s'), inplace=True)

        # --- ステップ3: マージ ---
        strain_aligned = raw_strain_series.reindex(df_tip.index, method='nearest', tolerance=pd.Timedelta(sensor_period_s, unit='s'))
        df_merged = df_tip.copy()
        df_merged['raw_strain'] = strain_aligned
        df_merged.dropna(subset=[TIP_X_COL_NAME, TIP_Y_COL_NAME, 'raw_strain'], inplace=True)
        if df_merged.empty:
            print("  -> [エラー] センサー出力と座標のマージに失敗しました。スキップします。")
            return

        # --- ステップ4: 距離(mm)を計算 ---
        coords = df_merged[[TIP_X_COL_NAME, TIP_Y_COL_NAME]].values
        start_coord = coords[0]
        effective_distance_px = np.sqrt(((coords - start_coord)**2).sum(axis=1))
        df_merged['effective_distance_mm'] = effective_distance_px / px_per_mm
        
        # --- ステップ5: グラフを描画 (生出力を使用) ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        time_axis = df_merged['time_s']
        
        # --- 上段 (距離グラフ) ---
        ax1.plot(time_axis, df_merged['effective_distance_mm'], color='green', label='Effective Distance', lw=2)
        ax1.set_ylabel('Distance (mm)') 
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='upper left')
        
        
        # --- 下段 (センサー生出力グラフ) ---
        ax2.plot(time_axis, df_merged['raw_strain'], color='red', label='Sensor Output', lw=2)
        ax2.set_ylabel('Sensor Output') 
        ax2.set_xlabel('Time (s)')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f"{base_name}_GUIDEWIRE_raw_strain_mm_graph.png") 
        plt.savefig(output_path)
        plt.close(fig)
        
        print(f"  -> グラフを保存しました: {os.path.basename(output_path)}")
        
    except Exception as e:
        print(f"  [!!! 重大なエラー !!!] {base_name} の処理中にエラーが発生しました: {e}")

# --- メイン実行ブロック ---
if __name__ == "__main__":
    
    # ... (前略: ユーザー入力とファイル紐付けロジック) ...
    
    print("--- v29.1: ガイドワイヤ解析 (生出力・mm単位) ---")
    print("把持力ファイルと先端座標ファイルを紐付けます。")
    print("="*50)

    # --- 1. ユーザー入力: フォルダパス ---
    force_dir = input(r"1. センサーCSVフォルダのパスを入力 (例: ...\時間同期_把持力_yuki_1): ").strip().replace('"', '')
    tip_dir = input(r"2. ガイドワイヤ先端座標CSVフォルダのパスを入力: ").strip().replace('"', '')
    output_dir = input(r"3. グラフ出力先フォルダのパスを入力: ").strip().replace('"', '')

    if not os.path.isdir(force_dir):
        print(f"[エラー] センサーCSVフォルダが見つかりません: {force_dir}")
        exit()
    if not os.path.isdir(tip_dir):
        print(f"[エラー] 先端座標フォルダが見つかりません: {tip_dir}")
        exit()
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 3. ファイルの紐付け ---
    force_files = sorted(glob.glob(os.path.join(force_dir, "*_trimmed.csv")))
    tip_files = sorted(glob.glob(os.path.join(tip_dir, "*_tips_coordinates.csv")))
    
    if len(force_files) != len(tip_files):
        print(f"[!!! 警告 !!!] ファイル数が一致しません！")
        print(f"  センサーファイル: {len(force_files)} 件")
        print(f"  先端座標ファイル: {len(tip_files)} 件")
        print("  処理を続行しますが、紐付けがズレる可能性があります。")

    def get_base_name(path, suffix):
        return os.path.basename(path).replace(suffix, "")

    print("\n--- 5. ファイル紐付けの確認 ---")
    file_map = dict(zip(force_files, tip_files))
    for i, (f_path, t_path) in enumerate(file_map.items()):
        f_base = get_base_name(f_path, "_trimmed.csv")
        t_base = get_base_name(t_path, "_tips_coordinates.csv")
        print(f"  [{i+1:02d}] {f_base}  <->  {t_base}")

    print("-" * 50)
    choice = input("この紐付けで解析を実行しますか？ (Yes: y / No: n) [Y/n]: ").strip().lower()
    
    if choice == 'n':
        print("処理を中断しました。")
        exit()

    # --- 6. 解析の実行 ---
    print("\n" + "="*50)
    print(f"--- 解析を実行します（全 {len(file_map)} 件） ---")
    
    for force_path, tip_path in file_map.items():
        analyze_guidewire_file(
            force_path, tip_path, output_dir,
            FPS_VIDEO, PIXELS_PER_MM 
        )

    print("\n" + "="*50)
    print(f"--- {os.path.basename(output_dir)} の解析が完了しました ---")