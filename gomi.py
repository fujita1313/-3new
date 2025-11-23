import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 解析パラメータ ---
FPS_VIDEO = 30.0
SENSOR_SAMPLE_MS = 10.0
DATA_START_ROW = 62
PIXELS_PER_MM = 4.6402

plt.rcParams.update({
    'font.size': 24, 
    'axes.labelsize': 24, 
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 20, # (凡例を追加するため v31.2 より変更)
    'figure.figsize': (15, 8), 
    'font.family': 'Arial'
})

# --- ★★★ v31.2_debug: デバッググラフ出力用の関数 ★★★ ---
def create_debug_plot(force_path, tip_path, a, b, col_c, col_d, output_dir, 
                        force_threshold, force_smoothing_window_ms):
    """
    v31.2_debug: 
    1試行分の「生データ」「平滑化データ」「閾値」を重ねてプロットし、PNGで保存する。
    """
    
    base_name = os.path.splitext(os.path.basename(tip_path))[0].replace("_tips_coordinates", "")
    print(f"--- デバッググラフを作成します: {base_name} ---")

    try:
        # --- ステップ1: 把持力読み込み & 平滑化 ---
        df_force = pd.read_csv(force_path, skiprows=DATA_START_ROW, header=None, encoding='cp932')
        strain_C = pd.to_numeric(df_force[col_c], errors='coerce')
        strain_D = pd.to_numeric(df_force[col_d], errors='coerce')
        if strain_C.isnull().all() or strain_D.isnull().all():
            print(f"   [エラー] ひずみ列が空です。")
            return
            
        strain_mean = (strain_C + strain_D) / 2
        force_mN = (a * strain_mean**2) + (b * strain_mean)
        force_N = force_mN / 1000.0
        force_N_offsetted = force_N - force_N.iloc[0]
        
        sensor_period_s = SENSOR_SAMPLE_MS / 1000.0
        time_s_force = df_force.index * sensor_period_s
        
        # 生データのSeries (10ms周期)
        force_series_raw = pd.Series(force_N_offsetted.values, index=pd.to_timedelta(time_s_force, unit='s'))

        # ★ 平滑化データのSeries ★
        window_size = int(force_smoothing_window_ms / SENSOR_SAMPLE_MS)
        if window_size < 1: window_size = 1
        force_series_smooth = force_series_raw.rolling(window=window_size, min_periods=1, center=True).mean()

        # --- ステップ2: 座標読み込み (時間軸のアラインにのみ使用) ---
        df_tip = pd.read_csv(tip_path)
        if df_tip.empty:
            print(f"   [エラー] 先端座標CSVが空です。")
            return
        df_tip['time_s'] = df_tip['frame'] / FPS_VIDEO
        df_tip.set_index(pd.to_timedelta(df_tip['time_s'], unit='s'), inplace=True)

        # --- ステップ3: マージ (30fpsのビデオ時間軸に合わせる) ---
        
        # 生データをアライン
        force_raw_aligned = force_series_raw.reindex(df_tip.index, method='nearest', tolerance=pd.Timedelta(sensor_period_s, unit='s'))
        # 平滑化データをアライン
        force_smooth_aligned = force_series_smooth.reindex(df_tip.index, method='nearest', tolerance=pd.Timedelta(sensor_period_s, unit='s'))
        
        df_merged = df_tip.copy()
        df_merged['force_raw'] = force_raw_aligned
        df_merged['force_smooth'] = force_smooth_aligned
        
        df_merged.dropna(subset=['force_raw', 'force_smooth'], inplace=True)
        if df_merged.empty:
            print(f"   [エラー] マージに失敗しました。")
            return

        # --- ステップ4: デバッググラフの作成 ---
        plt.figure()
        

        
        # 1. 平滑化後の波形 (v31.2 が判定に使うデータ)
        plt.plot(df_merged['time_s'], df_merged['force_smooth'], 
                 label=f'Smoothed ({int(force_smoothing_window_ms)}ms window)', 
                 color='blue', linewidth=2.5)
        
        # 2. 生の波形 (参考)
        plt.plot(df_merged['time_s'], df_merged['force_raw'], 
                 label='Raw Force Data (v31.1)', 
                 color='red', alpha=0.4, linestyle='--')
                 
        # 3. 閾値のライン
        plt.axhline(y=force_threshold, 
                    label=f'Threshold ({force_threshold} N)', 
                    color='black', linestyle=':', linewidth=2)
        
        plt.xlabel("Time (s)")
        plt.ylabel("Gripping Force (N)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # グラフのY軸の範囲を調整 (0N から ピークの1.2倍まで)
        max_val = df_merged['force_raw'].max()
        plt.ylim(bottom=0, top=max_val * 1.2)
        
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f"DEBUG_PLOT_{base_name}.png")
        plt.savefig(plot_path)
        plt.close()
        
        print(f"✓ デバッググラフを保存しました: {os.path.basename(plot_path)}")


    except Exception as e:
        print(f"     [!!! 重大なエラー !!!] {base_name} の処理中にエラーが発生しました: {e}")
        return

# --- ★★★ メイン実行ブロック (v31.2_debug) ★★★ ---
# --- ★★★ メイン実行ブロック (v31.2_debug v2: 「並び順」対応) ★★★ ---
if __name__ == "__main__":
    
    print("--- v31.2_debug v2: 平滑化デバッググラフ作成 (「並び順」対応) ---")
    print(" (1試行分のグラフを出力し、平滑化の影響を可視化します)")
    print("="*50)

    # --- 1. ユーザー入力: フォルダパス ---
    force_dir = input(r"1. 把持力CSVフォルダのパスを入力: ").strip().replace('"', '')
    tip_dir = input(r"2. 先端座標CSVフォルダのパスを入力: ").strip().replace('"', '')
    output_dir = input(r"3. グラフ出力先フォルダのパスを入力: ").strip().replace('"', '')

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
        a_coeff = float(input("   -> 係数 'a' (x^2): "))
        b_coeff = float(input("   -> 係数 'b' (x): "))
        col_c = int(input("   -> ひずみ列Cの番号 (0から, 例: 2 or 4): "))
        col_d = int(input("   -> ひずみ列Dの番号 (0から, 例: 3 or 5): "))
        
        print("\n--- 5. デバッグ対象の「先端座標ファイル」のベース名を入力してください ---")
        print(" (例: IMG_0627_trimmed)")
        target_base_name = input(r"   -> デバッグしたい試行のベース名: ").strip()

        print("\n--- 6. デバッグ用の「パラメータ」を入力してください ---")
        print(" (v31.2 で「0回」になった組み合わせを入力)")
        push_threshold = float(input("   -> 1. 閾値 (N) (例: 4.0): "))
        smoothing_ms = float(input("   -> 2. 移動平均の窓幅 (ms) (例: 100): "))
        
    except ValueError:
        print("[エラー] 無効な数値です。処理を終了します。")
        exit()

    # --- 7. ★ v2 修正: 「並び順」でのファイル紐付け ---
    print(f"\n--- 7. 「並び順」でファイルを紐付けます ---")
    
    # 1. 対象の先端座標ファイルのフルパスを作成
    target_tip_path = os.path.join(tip_dir, f"{target_base_name}_tips_coordinates.csv")
    if not os.path.exists(target_tip_path):
        print(f"[エラー] 対象の先端座標ファイルが見つかりません: {target_tip_path}")
        exit()

    # 2. 全ての先端座標ファイルを取得し、ソート
    all_tip_files = sorted(glob.glob(os.path.join(tip_dir, "*_tips_coordinates.csv")))
    
    # 3. 対象ファイルのインデックス(位置)を特定
    try:
        # (os.path.normpath でパスの区切り文字を統一してから比較)
        target_index = [os.path.normpath(p) for p in all_tip_files].index(os.path.normpath(target_tip_path))
    except ValueError:
        print(f"[エラー] ソートリストに対象ファイルが見つかりません: {target_tip_path}")
        exit()
        
    print(f"   -> 先端座標ファイル: {os.path.basename(target_tip_path)} (リストの {target_index + 1} 番目)")

    # 4. 全ての把持力ファイルを取得し、ソート
    all_force_files = sorted(glob.glob(os.path.join(force_dir, "*_trimmed.csv")))
    
    # 5. ファイル数が一致するか念のため確認
    if len(all_tip_files) != len(all_force_files):
        print(f"[!!! 警告 !!!] ファイル数が一致しません！ (把持力: {len(all_force_files)} vs 先端座標: {len(all_tip_files)})")
    
    # 6. 同じインデックスの把持力ファイルを取得
    try:
        target_force_path = all_force_files[target_index]
    except IndexError:
        print(f"[エラー] 把持力リストにインデックス {target_index} が存在しません。ファイル数がズレています。")
        exit()

    print(f"   -> 把持力ファイル: {os.path.basename(target_force_path)} (リストの {target_index + 1} 番目)")
    print(f"   (上記2つのファイルでデバッグを実行します)")
    
    # --- 8. デバッグ関数の実行 ---
    print(f"\n--- 8. デバッググラフを作成します ---")

    create_debug_plot(
        target_force_path,        # (v2 修正)
        target_tip_path,          # (v2 修正)
        a_coeff, b_coeff, col_c, col_d,
        output_dir,
        push_threshold,           
        smoothing_ms              
    )

    print("\n" + "="*50)
    print(f"--- デバッググラフの作成が完了しました ---")