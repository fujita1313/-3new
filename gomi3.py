import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# ***************************************************************
# *** ユーザー設定 ***
# ***************************************************************


# 解析対象のCSVファイル名
FILE_NAME = r"C:\Users\haruk\OneDrive\ドキュメント\修士\実験データ\20250806\解析結果\先端検出結果\2-1-1_trimmed_tip_coordinates.csv" 

# 動画のフレームレート (FPS)。時間軸の計算に必須です。
# 実際の動画のFPSに合わせてこの値を変更してください。
DUMMY_FPS = 30.0 

<<<<<<< HEAD

=======
>>>>>>> 885567797e074f0c87c65d9b088637d8eb17d717
# 平滑化ウィンドウサイズ (フレーム数)。W=5を指定。
SMOOTHING_WINDOW_SIZE = 5 

# 出力フォルダ名 (現在のディレクトリに作成されます)
OUTPUT_DIR = r"C:\Users\haruk\OneDrive\ドキュメント\修士\実験データ\20250806\解析結果"

# ***************************************************************

# Matplotlibの基本設定
plt.style.use('default') 
plt.rcParams.update({'figure.figsize': (10, 8)})

def plot_smoothed_coordinates(csv_path, fps, window_size, output_dir):
    """
    CSVファイルを読み込み、平滑化を適用し、元の座標と平滑化後の座標をプロットする。
    """
    print(f"Plotting coordinates from {os.path.basename(csv_path)}...")
    print(f"FPS: {fps}, Window Size: {window_size} を使用して平滑化を実行します。")
        
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"エラー: CSVファイルの読み込み中にエラーが発生しました: {e}")
        return

    base = os.path.splitext(os.path.basename(csv_path))[0]
    
    # --- データの前処理と平滑化 ---
    
    # 欠損値（'NaN'）を補完 (前方/後方補完)
    # 平滑化処理を安定させるため、前後の有効な値でNaNを埋めます。
    df['tip_x'] = df['tip_x'].fillna(method='ffill').fillna(method='bfill')
    df['tip_y'] = df['tip_y'].fillna(method='ffill').fillna(method='bfill')
    
    # 時間軸の計算 (フレーム / FPS)
    df['Time (s)'] = df['frame'] / fps
    
    # 単純移動平均 (SMA) による平滑化
    # center=Trueでウィンドウの中心に平均値を配置します。
    df['Smoothed X'] = df['tip_x'].rolling(window=window_size, center=True).mean()
    df['Smoothed Y'] = df['tip_y'].rolling(window=window_size, center=True).mean()
    
    # Rolling平均で生じるNaNを補完 (端の処理)
    df['Smoothed X'] = df['Smoothed X'].fillna(method='ffill').fillna(method='bfill')
    df['Smoothed Y'] = df['Smoothed Y'].fillna(method='ffill').fillna(method='bfill')

    # --- グラフの描画 ---
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True) # 上下2段のサブプロット
    
    # X座標 (上段)
    axes[0].plot(df['Time (s)'], df['tip_x'], color='gray', linestyle=':', label='Original X', alpha=0.9)
    axes[0].plot(df['Time (s)'], df['Smoothed X'], color='blue', label='Smoothed X')
    axes[0].set_ylabel('X coordinate (pixels)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # Y座標 (下段)
    axes[1].plot(df['Time (s)'], df['tip_y'], color='gray', linestyle=':', label='Original Y', alpha=0.9)
    axes[1].plot(df['Time (s)'], df['Smoothed Y'], color='red', label='Smoothed Y')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Y coordinate (pixels)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    # 全体のタイトル (オプション)
    fig.suptitle(f'{base} Coordinate Analysis (W={window_size}, FPS={fps})', fontsize=14)
    
    plt.tight_layout()
    
    # --- 画像の保存 ---
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{base}_smoothed_coordinates.png")
    plt.savefig(plot_path)
    plt.close(fig) 
    print(f"\nグラフを保存しました: {plot_path}")


if __name__ == "__main__":
    
    if not os.path.exists(FILE_NAME):
        print(f"エラー: 指定されたファイル '{FILE_NAME}' が見つかりません。ファイル名を確認してください。")
    else:
        plot_smoothed_coordinates(FILE_NAME, DUMMY_FPS, SMOOTHING_WINDOW_SIZE, OUTPUT_DIR)