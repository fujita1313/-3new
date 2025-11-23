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

# (v25.4 [cite: 35-46] より)
plt.rcParams.update({
    'font.size': 16, 
    'axes.labelsize': 22, 
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.figsize': (15, 10), # 2段グラフのため高さを変更
    'font.family': 'Arial'
})

def analyze_single_file(force_path, tip_path, output_dir, a, b, col_c, col_d, fps, vel_thresh_per_frame):
    """
    単一の「把持力CSV」と「先端座標CSV」を統合し、
    [画像 b25dbb.png] スタイルのグラフを生成する。
    """
    
    # ファイル名が異なるため、先端座標CSVのベース名を優先する
    base_name = os.path.splitext(os.path.basename(tip_path))[0].replace("_tips_coordinates", "")
    print(f"\n--- 処理中: {base_name} ---")

    try:
        # --- ステップ1: 把持力ファイル(入力A)を読み込み ---
        # (v_chap3 [cite: 34-42] と v25.4 [cite: 90-101] のロジックを統合)
        df_force = pd.read_csv(force_path, skiprows=DATA_START_ROW, header=None, encoding='cp932')
        strain_C = pd.to_numeric(df_force[col_c], errors='coerce')
        strain_D = pd.to_numeric(df_force[col_d], errors='coerce')
        if strain_C.isnull().all() or strain_D.isnull().all():
             print(f"  -> [エラー] ひずみ列 {col_c} または {col_d} が空です。スキップします。")
             return
        
        strain_mean = (strain_C + strain_D) / 2
        x = strain_mean - strain_mean.iloc[0] # 初期値をオフセット
        force_N = (a * x**2) + (b * x) # ★ 係数を適用 (mNではなくNとして扱う [cite: 37])
        
        sensor_period_s = SENSOR_SAMPLE_MS / 1000.0
        time_s_force = df_force.index * sensor_period_s
        force_series = pd.Series(force_N.values, index=pd.to_timedelta(time_s_force, unit='s'))

        # --- ステップ2: 先端座標ファイル(入力B)を読み込み ---
        # (v13.6 と v_chap3 [cite: 102-103] のロジック)
        df_tip = pd.read_csv(tip_path)
        if df_tip.empty or 'smooth_cath_x' not in df_tip.columns:
            print("  -> [エラー] 先端座標CSVが空か、'smooth_cath_x' 列がありません。スキップします。")
            return
            
        df_tip['time_s'] = df_tip['frame'] / fps
        df_tip.set_index(pd.to_timedelta(df_tip['time_s'], unit='s'), inplace=True)

        # --- ステップ3: データを時間軸でマージ ---
        # (v_chap3 [cite: 104-106] のロジック)
        force_aligned = force_series.reindex(df_tip.index, method='nearest', tolerance=pd.Timedelta(sensor_period_s, unit='s'))
        df_merged = df_tip.copy()
        df_merged['force'] = force_aligned
        
        # 解析に必要なデータが揃っている行のみを対象
        df_merged.dropna(subset=['smooth_cath_x', 'smooth_cath_y', 'force'], inplace=True)
        if df_merged.empty:
            print("  -> [エラー] 把持力と座標のマージに失敗しました。スキップします。")
            return

        # --- ステップ4: 距離と動的/静的状態を計算 ---
        # (v_chap3 [cite: 110-113] と chap3.pdf [cite: 335-337] のロジック)
        coords = df_merged[['smooth_cath_x', 'smooth_cath_y']].values
        diffs = np.diff(coords, axis=0)
        step_distances = np.sqrt((diffs**2).sum(axis=1))
        
        # 累積距離
        df_merged['cumulative_distance'] = np.insert(step_distances, 0, 0).cumsum()
        # フレーム間速度 (pixels / frame)
        df_merged['velocity_px_f'] = np.insert(step_distances, 0, 0)
        # 静的状態か？ (True/False)
        df_merged['is_static'] = df_merged['velocity_px_f'] <= vel_thresh_per_frame

        # --- ステップ5: グラフを描画 ---
        # (image_b25dbb.png  のスタイル)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        time_axis = df_merged['time_s']
        
        # --- 上段 (距離グラフ) ---
        ax1.plot(time_axis, df_merged['cumulative_distance'], color='blue', label='Cumulative Distance')
        ax1.set_ylabel('Distance (pixels)')
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # Y軸の範囲を取得
        y_min, y_max = ax1.get_ylim()

        # ★ 動的/静的の背景色をプロット
        # (fill_between を使い、is_static が True の領域をグレーで塗りつぶす)
        ax1.fill_between(time_axis, y_min, y_max, 
                         where=df_merged['is_static'], 
                         facecolor='grey', alpha=0.2, 
                         label='Static State (v < 35 pps)')
        
        ax1.set_ylim(y_min, y_max) # fill_between で変わった可能性のあるY軸を戻す
        ax1.legend(loc='upper left')

        # --- 下段 (把持力グラフ) ---
        ax2.plot(time_axis, df_merged['force'], color='red', label='Gripping Force')
        ax2.set_ylabel('Gripping Force (N)')
        ax2.set_xlabel('Time (s)')
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        # Y軸の最小値を0に固定
        y_min, y_max = ax2.get_ylim()
        ax2.set_ylim(0, y_max * 1.1) # 最小値を0に
        ax2.legend(loc='upper right')
        
        # --- 共通設定 ---
        plt.tight_layout()
        
        # グラフを保存
        output_path = os.path.join(output_dir, f"{base_name}_force_distance_graph.png")
        plt.savefig(output_path)
        plt.close(fig)
        
        print(f"  -> グラフを保存しました: {os.path.basename(output_path)}")
        
    except Exception as e:
        print(f"  [!!! 重大なエラー !!!] {base_name} の処理中にエラーが発生しました: {e}")

# --- メイン実行ブロック ---
if __name__ == "__main__":
    
    print("--- v28: Task 1 (S-Curve) 統合解析スクリプト ---")
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
    # (v_chap3 [cite: 20-27] を参考に、被験者ごとに設定)
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
    # (v25.4 [cite: 110-132] のロジック)
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

    # ファイル名のベースを抽出し、比較・確認する
    # (例: '2025_1023_111419_357' と 'IMG_0645_trimmed')
    def get_base_name(path, suffix):
        return os.path.basename(path).replace(suffix, "")

    print("\n--- 5. ファイル紐付けの確認 ---")
    print(f"'{os.path.basename(force_dir)}' と '{os.path.basename(tip_dir)}' のファイルを順序で紐付けます。")
    
    file_map = dict(zip(force_files, tip_files))
    is_mapping_ok = True
    for i, (f_path, t_path) in enumerate(file_map.items()):
        f_base = get_base_name(f_path, "_trimmed.csv")
        t_base = get_base_name(t_path, "_tips_coordinates.csv")
        print(f"  [{i+1:02d}] {f_base}  <->  {t_base}")
        # (ファイル名が完全に異なるため、自動チェックはせず表示のみ)

    print("-" * 50)
    choice = input("この紐付けで解析を実行しますか？ (Yes: y / No: n) [Y/n]: ").strip().lower()
    
    if choice == 'n':
        print("処理を中断しました。")
        exit()

    # --- 6. 解析の実行 ---
    print("\n" + "="*50)
    print(f"--- 解析を実行します（全 {len(file_map)} 件） ---")
    
    # 解析用パラメータを計算
    vel_thresh_per_frame = VELOCITY_THRESHOLD_PPS / FPS_VIDEO

    for force_path, tip_path in file_map.items():
        analyze_single_file(
            force_path, tip_path, output_dir,
            a_coeff, b_coeff, col_c, col_d,
            FPS_VIDEO, vel_thresh_per_frame
        )

    print("\n" + "="*50)
    print(f"--- {os.path.basename(output_dir)} の解析が完了しました ---")