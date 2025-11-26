import pandas as pd
import os
import glob

# --- ユーザー設定 ---

# 1. 電圧がONと判定される閾値 (V)
VOLTAGE_THRESHOLD = 1.0

# 2. WAVE LOGGERのCSVファイルで、実際のデータが始まる行番号
#    (0から数えるため、63行目であれば「62」と指定)
DATA_START_ROW = 62

# 3. 電圧データが記録されている列番号
#    (0から数えるため、10列目であれば「9」と指定)
VOLTAGE_COLUMN_INDEX = 9 
 
# --- ここまで ---


def trim_wavelogger_csv(filepath, output_dir): # ★ 修正: file_index を削除
    """
    単一のWAVE LOGGER CSVファイルを処理し、有効区間を切り出して保存する。
    """
    # ★★★ 修正点: 親フォルダの名前を取得 ★★★
    parent_folder_name = os.path.basename(os.path.dirname(filepath))
    print(f"--- 処理開始: {parent_folder_name} ---")

    try:
        # --- ステップ1: ヘッダーとデータを別々に読み込む ---
        with open(filepath, 'r', encoding='cp932') as f:
            header_lines = [next(f) for _ in range(DATA_START_ROW)]
        df = pd.read_csv(filepath, skiprows=DATA_START_ROW, header=None, encoding='cp932')

        if VOLTAGE_COLUMN_INDEX >= len(df.columns):
            print(f"  [エラー] 電圧列 {VOLTAGE_COLUMN_INDEX} が見つかりません。スキップします。")
            return
        voltages = pd.to_numeric(df[VOLTAGE_COLUMN_INDEX], errors='coerce').fillna(0)

        # --- ステップ2: ONイベントとOFFイベントのインデックスを探す ---
        on_index = -1
        off_index = -1
        for i, voltage in enumerate(voltages):
            if voltage >= VOLTAGE_THRESHOLD:
                on_index = i
                print(f"  ONイベントを発見 (行: {DATA_START_ROW + on_index})")
                break
        if on_index == -1:
            print(f"  [警告] ONイベントが見つかりませんでした。")
            return
        for i in range(on_index, len(voltages)):
            if voltages.iloc[i] < VOLTAGE_THRESHOLD:
                off_index = i
                print(f"  OFFイベントを発見 (行: {DATA_START_ROW + off_index})")
                break
        if off_index == -1:
            off_index = len(df) -1
            print(f"  [情報] OFFイベントが見つかりませんでした。ファイルの末尾までとします。")

        # --- ステップ3: データを切り出して新しいCSVファイルを作成 ---
        trimmed_df = df.iloc[on_index : off_index + 1]

        # ★★★ 修正点: ファイル名を親フォルダ名(タイムスタンプ)に変更 ★★★
        output_filename = f"{parent_folder_name}_trimmed.csv"
        output_filepath = os.path.join(output_dir, output_filename)

        with open(output_filepath, 'w', encoding='cp932', newline='') as f:
            f.writelines(header_lines)
            trimmed_df.to_csv(f, header=False, index=False)
        
        print(f"  [成功] 切り出したファイルを保存しました: {output_filepath}")

    except FileNotFoundError:
        print(f"  [エラー] ファイルが見つかりません: {filepath}")
    except pd.errors.EmptyDataError:
        print(f"  [エラー] ファイルは空です: {filepath}")
    except Exception as e:
        print(f"  [エラー] 予期せぬエラーが発生しました: {e}")


if __name__ == "__main__":
    input_folder = input("解析対象の『親』フォルダのパスを入力してください (例: .../wavelogger/ガイドワイヤ): ").strip().replace('"', '')
    if not os.path.isdir(input_folder):
        print("エラー: 有効なフォルダパスではありません。")
    else:
        output_folder = input("切り出したファイルの保存先フォルダのパスを入力してください: ").strip().replace('"', '')
        if not os.path.isdir(output_folder):
            print("エラー: 有効な保存先フォルダパスではありません。作成します。")
            os.makedirs(output_folder, exist_ok=True)

        search_pattern = os.path.join(input_folder, "**", "auto$0.csv")
        file_list = sorted(glob.glob(search_pattern, recursive=True))

        if not file_list:
            print(f"指定されたフォルダ '{input_folder}' (サブフォルダ含む) 内に 'auto$0.csv' が見つかりませんでした。")
        else:
            print(f"\n{len(file_list)}個のファイルが見つかりました。処理を開始します。")
            
            # ★★★ 修正点: enumerate を削除 ★★★
            for file_path in file_list:
                trim_wavelogger_csv(file_path, output_folder) # 連番 i を渡すのをやめる
                
            print("\n全ての処理が完了しました。")