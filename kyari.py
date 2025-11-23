import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from sklearn.metrics import r2_score, mean_squared_error


# --- Matplotlibの全体的なスタイルとフォントサイズを設定 ---
plt.rcParams.update({
    'font.size': 16,            # 全体の基本フォントサイズ
    'axes.titlesize': 22,       # グラフタイトルのフォントサイズ
    'axes.labelsize': 20,       # 軸ラベルのフォントサイズ
    'xtick.labelsize': 14,      # X軸目盛りのフォントサイズ
    'ytick.labelsize': 14,      # Y軸目盛りのフォントサイズ
    'legend.fontsize': 16,      # 凡例のフォントサイズ
    'figure.figsize': (10, 7),  # グラフのサイズ (inch)
    'figure.dpi': 100,          # 解像度
    'font.family': 'Arial'      # 可読性の高いフォントを指定
})

# --- 1. ★★★ 設定項目 ★★★ ---
# ★ 修正点 1: 入力パスを '.../20251022/wavelogger/ガイドワイヤ' に変更
base_data_dir = r"C:\Users\haruk\OneDrive\ドキュメント\修士\実験データ\20251022\wavelogger\1023_tomohiro_3"
output_dir = r"C:\Users\haruk\OneDrive\ドキュメント\修士\実験データ\20251022\解析結果\カテーテル\1023_tomohiro_3"
os.makedirs(output_dir, exist_ok=True)

# ★ 修正点 2: WaveLoggerの列定義 (右手用)
# chC(ひずみ1)=2, chD(ひずみ2)=3
# chG(ロードセルX)=6, chH(ロードセルY)=7, chI(ロードセルZ)=8
use_cols = {
    4: 'strain_C', 5: 'strain_D', 
    6: 'load_G', 7: 'load_H', 8: 'load_I'
}
# ★ WaveLoggerのヘッダー行数
DATA_START_ROW = 62

# --- 2. 解析処理 ---
target_files = glob.glob(os.path.join(base_data_dir, '**', 'auto$0.csv'), recursive=True)
print(f"✅ {len(target_files)}個のデータファイルを検出しました。解析を開始します...")
all_results = []

for file_path in target_files:
    try:
        # 親フォルダ名 (例: 2025_1022_160651_856) をデータセット名とする
        dataset_name = os.path.basename(os.path.dirname(file_path))
        print(f"\n--- [{dataset_name}] の解析中 ---")

        # ★ 修正点 3: 'migi'/'hidari' の分岐ロジックを削除 ---
        # (if 'migi' in dataset_name: ... else: ... を削除)
        # 共通の use_cols を使用します。
        
        df = pd.read_csv(file_path, skiprows=DATA_START_ROW, header=None, encoding='cp932')
        
        # 必要な列だけを抽出
        # 列が存在するかチェック
        missing_cols = [c for c in use_cols.keys() if c >= len(df.columns)]
        if missing_cols:
            print(f"警告: 必要な列 {missing_cols} がファイルに存在しません (列数: {len(df.columns)})。スキップします。")
            continue
            
        df = df[list(use_cols.keys())]
        df.columns = list(use_cols.values())
        df = df.apply(pd.to_numeric, errors='coerce').dropna()

        if df.empty:
            print(f"警告: データが空です。スキップします。")
            continue
        
        # --- データ前処理とオフセット ---
        df['load_I'] /= 11.7
        strain_gauge_mean = df[['strain_C', 'strain_D']].mean(axis=1)
        
        coefficient_index = np.array([[0.05548, 0.00013, 0.00112], [0.00050, 0.05422, -0.00034], [-0.00088, 0.00038, 0.05850]])
        loadcell_raw = df[['load_G', 'load_H', 'load_I']].values
        loadcell_c = np.copy(loadcell_raw)
        loadcell_c[:, [0, 1]] *= 100; loadcell_c[:, 2] *= 200
        mod_loadcell = (coefficient_index @ loadcell_c.T).T
        force_z_axis_mn = mod_loadcell[:, 2] * 1000
        
        force_offsetted = force_z_axis_mn - force_z_axis_mn[0]
        strain_offsetted = strain_gauge_mean - strain_gauge_mean.iloc[0]

        df_processed = pd.DataFrame({'Strain': strain_offsetted, 'Force_Z': force_offsetted})
        
        # ★ 修正点 4: トライアル分割(trial_id)のロジックを変更 ---
        # (v18コードの trial_id ロジックは、トリム前のデータ用でした)
        # (v6コードのロジックを採用し、「押し込み区間」だけを抽出します)
        all_rising_data = df_processed[
            (df_processed['Force_Z'].diff().fillna(0) >= 0) &
            (df_processed['Force_Z'] > 0)
        ]
        
        if all_rising_data.empty:
            print(f"警告: 解析対象のデータ区間(押し込み区間)が見つかりませんでした。スキップします。")
            continue
        
        # --- モデル作成と精度評価 (0-10Nでフィッティング) ---
        df_full_range = all_rising_data[all_rising_data['Force_Z'] <= 10000] # 10N
        x_full_all, y_full_all = df_full_range['Strain'].values, df_full_range['Force_Z'].values
        
        if len(x_full_all) < 10:
            print(f"警告: 0-10Nのデータが不足。スキップします。")
            continue

        A_full = np.vstack([x_full_all**2, x_full_all]).T
        a_model, b_model = np.linalg.lstsq(A_full, y_full_all, rcond=None)[0]
        print(f"  > 作成モデル: y = {a_model:.4f}x² + {b_model:.4f}x")

        # 0-6N範囲での精度
        df_low = all_rising_data[all_rising_data['Force_Z'] <= 6000] # 6N
        x_low_all, y_low_all = df_low['Strain'].values, df_low['Force_Z'].values
        if len(x_low_all) > 0:
            y_pred_low = a_model * x_low_all**2 + b_model * x_low_all
            r2_low, rmse_low = r2_score(y_low_all, y_pred_low), np.sqrt(mean_squared_error(y_low_all, y_pred_low))
            print(f"  > 精度 (0-6N): R²={r2_low:.4f}, RMSE={rmse_low:.2f} mN")
        else:
            r2_low, rmse_low = np.nan, np.nan
            print(f"  > 精度 (0-6N): データなし")

        # 0-10N範囲での精度
        y_pred_full = a_model * x_full_all**2 + b_model * x_full_all
        r2_full, rmse_full = r2_score(y_full_all, y_pred_full), np.sqrt(mean_squared_error(y_full_all, y_pred_full))
        print(f"  > 精度 (0-10N): R²={r2_full:.4f}, RMSE={rmse_full:.2f} mN")

        all_results.append({
            'dataset': dataset_name, 'model_equation': f'y = {a_model:.4f}x^2 + {b_model:.4f}x',
            'R2 (0-6N)': r2_low, 'RMSE_0-6N (mN)': rmse_low,
            'R2 (0-10N)': r2_full, 'RMSE_0-10N (mN)': rmse_full
        })
        
        # --- グラフ描画と保存 ---
        fig, ax = plt.subplots() 
        ax.scatter(x_full_all, y_full_all, s=20, alpha=0.6, label='Measured Data')
        
        x_max_plot = x_full_all.max() * 1.05
        x_range_plot = np.linspace(0, x_max_plot, 200)
        y_range_plot = a_model * x_range_plot**2 + b_model * x_range_plot
        
        formula = f'$y = {a_model:.3f}x^2 + {b_model:.3f}x$'
        ax.plot(x_range_plot, y_range_plot, color='red', lw=2.5, label=formula)
        
        ax.set_xlabel('Sensor Output (offset)') 
        ax.set_ylabel('Gripping Force (mN)')
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plot_filename = f"{dataset_name}_plot.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"  > グラフを保存しました: {plot_filename}")

    except Exception as e:
        print(f"エラー: [{dataset_name}] 処理中に予期せぬ問題: {e}")

# --- 3. 全結果をExcelに保存 ---
if all_results:
    df_summary = pd.DataFrame(all_results)
    summary_path = os.path.join(output_dir, 'calibration_summary_10N_fit.xlsx')
    df_summary.to_excel(summary_path, index=False)
    print(f"\n\n✨ 全ての解析が完了しました！ ✨")
    print(f"結果のサマリーをExcelに保存しました: {summary_path}")
else:
    print("\n\n解析できる有効なデータがありませんでした。")