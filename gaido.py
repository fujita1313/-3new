import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re 


# --- 1. ★★★ ユーザー設定 ★★★ ---
FORCE_CSV_DIR = r"C:\Users\haruk\OneDrive\ドキュメント\修士\実験データ\20251022\解析結果\ガイドワイヤ\時間同期-把持力"
COLLISION_CSV_DIR = r"C:\Users\haruk\OneDrive\ドキュメント\修士\実験データ\20251022\解析結果\ガイドワイヤ\先端検出結果_v23_CollisionZone"
OUTPUT_DIR = r"C:\Users\haruk\OneDrive\ドキュメント\修士\実験データ\20251022\解析結果\ガイドワイヤ\統合解析_v25_Time"

# --- 2. CSV読み込み設定 ---
DATA_START_ROW = 62
STRAIN_C_COL = 2
STRAIN_D_COL = 3

# --- 3. ★★★ 時間軸パラメータ (New) ★★★ ---
SENSOR_SAMPLE_MS = 10.0
FPS_VIDEO = 30.0

# --- 4. Matplotlibスタイル設定 ---
# ★★★ 修正: 軸タイトル、目盛り、凡例を大幅に拡大 ★★★
plt.rcParams.update({
    'font.size': 20, # 基本フォント
    'axes.labelsize': 36, # 軸タイトルサイズ (大きく)
    'xtick.labelsize': 24, # X軸目盛りサイズ (大きく)
    'ytick.labelsize': 24, # Y軸目盛りサイズ (大きく)
    'legend.fontsize': 24, # 凡例サイズ (大きく)
    'figure.figsize': (15, 7), 
    'font.family': 'Arial'
})

# --- (v25 と同じ関数群: parse_indices, setup_coefficients) ---
def parse_indices(index_str, max_index):
    """ "1-5, 7" などを [0, 1, 2, 3, 4, 6] に変換 """
    indices = set()
    parts = index_str.split(',')
    for part in parts:
        part = part.strip()
        if not part: continue
        if '-' in part:
            start, end = part.split('-')
            try:
                start_idx = int(start) - 1
                end_idx = int(end) - 1 
                if start_idx < 0 or end_idx >= max_index: raise ValueError("範囲外")
                for i in range(start_idx, end_idx + 1):
                    indices.add(i)
            except ValueError:
                print(f"  -> 警告: 無効な範囲 '{part}'。")
        else:
            try:
                idx = int(part) - 1
                if 0 <= idx < max_index:
                    indices.add(idx)
                else:
                    print(f"  -> 警告: 無効な番号 '{part}'。")
            except ValueError:
                print(f"  -> 警告: 無効な入力 '{part}'。")
    return sorted(list(indices))

def setup_coefficients(file_list):
    """ インタラクティブに係数マップを作成する """
    print("\n--- フェーズA: キャリブレーション係数の設定 ---")
    print("解析対象の把持力ファイルリスト:")
    for i, file_path in enumerate(file_list):
        print(f"  [{i+1:02d}] {os.path.basename(file_path)}")
    print("-" * 50)
    
    coeff_map = {} 
    group_num = 1
    
    while True:
        print(f"\n[係数グループ {group_num}]")
        index_str = input(f"  -> このグループを適用するファイル番号を入力 (例: 1-5, 8) (完了の場合はEnter): ").strip()
        if not index_str:
            break
            
        try:
            a_coeff = float(input(f"  -> グループ {group_num} の係数 'a' (x^2) を入力: "))
            b_coeff = float(input(f"  -> グループ {group_num} の係数 'b' (x) を入力: "))
        except ValueError:
            print("  -> エラー: 有効な数値を入力してください。")
            continue
            
        indices_to_apply = parse_indices(index_str, len(file_list))
        
        if not indices_to_apply:
            print("  -> 有効なファイルが指定されませんでした。")
            continue
            
        print(f"  -> 以下のファイルに F = {a_coeff:.4f}x^2 + {b_coeff:.4f}x を適用します:")
        for idx in indices_to_apply:
            file_path = file_list[idx]
            coeff_map[file_path] = {'a': a_coeff, 'b': b_coeff}
            print(f"     [{idx+1:02d}] {os.path.basename(file_path)}")
            
        group_num += 1
        
    print("\n--- 係数の設定が完了しました ---")
    return coeff_map

# --- 5. メイン処理 (★ v25.5 修正) ---
def analyze_single_file(force_path, collision_path, output_dir, a_coeff, b_coeff):
    """
    v25.5: N単位に修正、プロット線太く、軸タイトル/凡例特大化
    """
    base_name_force = os.path.splitext(os.path.basename(force_path))[0]
    base_name_coll = os.path.splitext(os.path.basename(collision_path))[0]
    
    print(f"\n--- 処理中: {base_name_force} (把持力)")
    print(f"           + {base_name_coll} (衝突)")
    print(f"  (使用係数: a={a_coeff:.4f}, b={b_coeff:.4f})")

    # --- ステップ1: 衝突ファイル(入力B)を読み込み、★時間★ を計算 ---
    collision_time_s = -1.0 
    if not os.path.exists(collision_path):
        print(f"  -> 警告: 衝突ファイルが見つかりません。スキップします。\n     {collision_path}")
        return
        
    try:
        df_collision = pd.read_csv(collision_path)
        collision_frames = df_collision[df_collision['is_collided'] == 1]['frame']
        
        if not collision_frames.empty:
            collision_frame = collision_frames.iloc[0]
            collision_time_s = collision_frame / FPS_VIDEO 
            print(f"  -> 衝突フレームを検出: {collision_frame} (={collision_time_s:.3f} s)")
        else:
            print(f"  -> 衝突は検出しませんでした。")

    except Exception as e:
        print(f"  -> エラー: 衝突ファイル {os.path.basename(collision_path)} の読み込みに失敗: {e}")
        return

    # --- ステップ2: 把持力ファイル(入力A)を読み込み、★時間軸★ で計算 ---
    try:
        df_force = pd.read_csv(force_path, skiprows=DATA_START_ROW, header=None, encoding='cp932')
        
        strain_C = pd.to_numeric(df_force[STRAIN_C_COL], errors='coerce')
        strain_D = pd.to_numeric(df_force[STRAIN_D_COL], errors='coerce')
        
        if strain_C.isnull().all() or strain_D.isnull().all():
             print(f"  -> エラー: ひずみ列 {STRAIN_C_COL} または {STRAIN_D_COL} が空です。")
             return
             
        strain_mean = (strain_C + strain_D) / 2
        
        x = strain_mean - strain_mean.iloc[0]
        force_mN = (a_coeff * x**2) + (b_coeff * x)
        
        # ★★★ 修正: 単位をNに変換 ★★★
        force_N = force_mN / 1000.0 
        
        sensor_period_s = SENSOR_SAMPLE_MS / 1000.0
        time_s_force = df_force.index * sensor_period_s

        # --- ステップ3: グラフを描画 ★★★ 修正箇所 ★★★ ---
        plt.figure() 
        
        # ★ 1. プロット線の太さを lw=3 に変更
        plt.plot(time_s_force, force_N, label='Calculated Gripping Force (N)', color='blue', lw=3) 
        
        if collision_time_s != -1.0:
            plt.axvline(collision_time_s, color='red', linestyle='--', linewidth=3, # linewidth=3 に変更
                                 label=f'Collision Detected ({collision_time_s:.3f} s)')
        
        # ★ 2. タイトルを削除 (rcParamsでaxes.labelsize=36が適用される)
        # plt.title(f'Gripping Force Analysis - {base_name_force}') 
        
        # ★ 3. 軸ラベルの単位を N に修正 (fontsizeはrcParams=36を適用)
        plt.xlabel('Time (s)') 
        plt.ylabel('Gripping Force (N)')
        
        # ★ 4. 凡例のフォントサイズは rcParams (24) を使用
        plt.legend() 
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        plot_path = os.path.join(OUTPUT_DIR, f"{base_name_force}_force_plot_N.png") # ファイル名もNに変更
        plt.savefig(plot_path)
        plt.close() 
        print(f"  -> グラフを保存しました: {plot_path}")

    except Exception as e:
        print(f"  -> エラー: 把持力ファイル {os.path.basename(force_path)} の処理に失敗: {e}")

# --- 6. 実行ブロック (v25.1 と同じ) ---
if __name__ == "__main__":
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    search_pattern_A = os.path.join(FORCE_CSV_DIR, "*_trimmed.csv")
    force_files = sorted(glob.glob(search_pattern_A))

    search_pattern_B = os.path.join(COLLISION_CSV_DIR, "*_collision_status.csv")
    collision_files = sorted(glob.glob(search_pattern_B))

    if not force_files:
        print(f"エラー: {FORCE_CSV_DIR} に ..._trimmed.csv が見つかりません。")
        exit()
    if not collision_files:
        print(f"エラー: {COLLISION_CSV_DIR} に ..._collision_status.csv が見つかりません。")
        exit()

    if len(force_files) != len(collision_files):
        print(f"エラー: ファイル数が一致しません！")
        print(f"  把持力ファイル (A): {len(force_files)} 件")
        print(f"  衝突ファイル (B): {len(collision_files)} 件")
        print("フォルダ内を確認してください。処理を中断します。")
        exit()
        
    file_map = dict(zip(force_files, collision_files))
    
    print(f"--- {len(force_files)} 件のファイルペアを紐付けました ---")
    for f_path, c_path in file_map.items():
        print(f"  [Force] {os.path.basename(f_path)} -> [Collision] {os.path.basename(c_path)}")
    print("="*50)
    
    coeff_map = setup_coefficients(force_files)

    if not coeff_map:
        print("係数が何も設定されなかったため、処理を終了します。")
        exit()

    print("\n" + "="*50)
    print("--- フェーズB: 統合解析を実行します ---")

    for force_path in coeff_map.keys():
        
        coeffs = coeff_map[force_path]
        a_coeff = coeffs['a']
        b_coeff = coeffs['b']
        
        collision_path = file_map[force_path] 

        analyze_single_file(force_path, collision_path, OUTPUT_DIR, a_coeff, b_coeff)
            
    print("\n--- 全ての処理が完了しました ---")