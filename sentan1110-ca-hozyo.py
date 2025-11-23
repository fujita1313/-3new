import cv2
import numpy as np
import os
import pandas as pd
import glob
from collections import deque
import time 
import matplotlib.pyplot as plt

# --- ★★★ ユーザー設定 (v13.8) ★★★ ---
# (v13.6/v13.7 と設定は同じ)

# 1. 解析対象の「動画フォルダ」
INPUT_DIR = r"C:\Users\haruk\OneDrive\ドキュメント\修士\実験データ\20251022\解析結果\時間同期\カテーテル\trimmed_videos"
# 2. 解析結果の「出力先フォルダ」
OUTPUT_DIR = r"C:\Users\haruk\OneDrive\ドキュメント\修士\実験データ\20251022\解析結果\カテーテル\先端検出結果_v13.6_Debug"
# 3. 解析対象の動画の拡張子
VIDEO_EXTENSION = "_trimmed.avi"

# 4. v10.5 アルゴリズム設定
BIN_THRESHOLD = 0 
THICKNESS_THRESHOLD = 1.3
LOOKAHEAD_WINDOW = 25
MIN_IN_WINDOW = 13 
CLOSING_ITERATIONS = 1

# 5. 平滑化の設定
MOVING_AVERAGE_WINDOW = 5 

# 7. プレビュー動画の作成有無
CREATE_PREVIEW_VIDEO = True 

# 8. 平滑化「検証グラフ」の作成有無
CREATE_VERIFICATION_GRAPH = True

# --- 設定ここまで ---


# --- ユーティリティ関数群 (v13.7 と変更なし) ---
def select_analysis_roi(video_path):
    """ROIを1つ選択させる"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        print(f"  [エラー] 動画 {video_path} が開けません。")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()
    if not ret: 
        print(f"  [エラー] ROI選択用のフレーム読み込み失敗 (Frame: {target_frame})")
        return None
    print("  --- ROIの選択 ---")
    print("  解析したい領域をドラッグして囲み、[Enter]キーを押してください。")
    print("  (キャンセルする場合は [c] キーを押してください)")
    h, w, _ = frame.shape
    scale = min(1.0, 1200 / w)
    preview_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    window_title = f"Select Analysis Area for {os.path.basename(video_path)} (Press Enter)"
    roi_preview = cv2.selectROI(window_title, preview_frame, False)
    cv2.destroyWindow(window_title)
    roi = tuple(int(c / scale) for c in roi_preview)
    if roi[2] == 0 or roi[3] == 0:
        print("  -> ROI選択がキャンセルされました。")
        return None
    print(f"  -> ROI {roi} が選択されました。")
    return roi

def get_main_body_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask)
    main_contour = max(contours, key=cv2.contourArea)
    main_body_mask = np.zeros_like(mask)
    cv2.drawContours(main_body_mask, [main_contour], -1, 255, thickness=cv2.FILLED)
    return main_body_mask

def find_endpoints(thinned_image):
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    thinned_01 = thinned_image // 255
    neighbor_count = cv2.filter2D(thinned_01, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    endpoint_mask = ((thinned_01 == 1) & (neighbor_count == 1))
    rows, cols = np.where(endpoint_mask)
    return list(zip(cols, rows)) 

def get_endpoint_by_x(endpoints, find_min=True):
    if not endpoints:
        return None
    if find_min:
        return min(endpoints, key=lambda p: p[0])
    else:
        return max(endpoints, key=lambda p: p[0])

def trace_skeleton_and_get_thickness_and_coords(skeleton, dist_map, start_point):
    if start_point is None:
        return []
    h, w = skeleton.shape
    queue = deque([(start_point[1], start_point[0], 0.0)]) 
    visited = np.zeros((h, w), dtype=bool) 
    visited[start_point[1], start_point[0]] = True 
    results = [] 
    neighbors = [(-1, -1), (-1, 0), (-1, 1), 
                 ( 0, -1),          ( 0, 1),
                 ( 1, -1), ( 1, 0), ( 1, 1)]
    distances = [np.sqrt(dy**2 + dx**2) for dy, dx in neighbors]
    while queue:
        y, x, dist = queue.popleft() 
        current_thickness = dist_map[y, x]
        results.append( (dist, current_thickness, x, y) ) 
        for i in range(8): 
            dy, dx = neighbors[i]
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                if skeleton[ny, nx] == 255 and not visited[ny, nx]:
                    visited[ny, nx] = True 
                    new_dist = dist + distances[i] 
                    queue.append((ny, nx, new_dist)) 
    results.sort(key=lambda item: item[0])
    return results

def find_junction_with_lookahead(trace_data, threshold, window_size, min_in_window):
    if not trace_data:
        return None, None
    n_points = len(trace_data)
    raw_thickness_values = np.array([item[1] for item in trace_data])
    above_thresh = raw_thickness_values > threshold
    diff = np.diff(above_thresh.astype(int), prepend=False) 
    start_indices = np.where(diff == 1)[0] + 1
    if len(start_indices) == 0:
         if above_thresh[0]: 
             return (trace_data[0][2], trace_data[0][3]), trace_data[0][0]
         return None, None
    for start_index in start_indices:
        end_index = min(start_index + window_size, n_points)
        window_data = raw_thickness_values[start_index:end_index]
        count_above_thresh = np.sum(window_data > threshold)
        if count_above_thresh >= min_in_window:
            junction_point = (trace_data[start_index][2], trace_data[start_index][3])
            junction_dist = trace_data[start_index][0]
            return junction_point, junction_dist
    return None, None

def plot_smoothing_comparison(df, output_path):
    """(v13.6 と変更なし)"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    time_axis = df['time_s']
    ax1.plot(time_axis, df['catheter_tip_x'], color='grey', linestyle=':', alpha=0.7, label='Original Cath X')
    ax1.plot(time_axis, df['smooth_cath_x'], color='blue', label=f'Smoothed Cath X (w={MOVING_AVERAGE_WINDOW})')
    ax1.plot(time_axis, df['catheter_tip_y'], color='silver', linestyle=':', alpha=0.7, label='Original Cath Y')
    ax1.plot(time_axis, df['smooth_cath_y'], color='red', label=f'Smoothed Cath Y (w={MOVING_AVERAGE_WINDOW})')
    ax1.set_ylabel('Catheter Coords (pixels)', fontsize=20)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=12)
    ax1.tick_params(axis='y', labelsize=14)
    ax1.set_title(f'Smoothing Verification (Window = {MOVING_AVERAGE_WINDOW})', fontsize=22)
    ax2.plot(time_axis, df['guidewire_tip_x'], color='grey', linestyle=':', alpha=0.7, label='Original GW X')
    ax2.plot(time_axis, df['smooth_gw_x'], color='blue', label=f'Smoothed GW X (w={MOVING_AVERAGE_WINDOW})')
    ax2.plot(time_axis, df['guidewire_tip_y'], color='silver', linestyle=':', alpha=0.7, label='Original GW Y')
    ax2.plot(time_axis, df['smooth_gw_y'], color='red', label=f'Smoothed GW Y (w={MOVING_AVERAGE_WINDOW})')
    ax2.set_ylabel('Guidewire Coords (pixels)', fontsize=20)
    ax2.set_xlabel('Time (s)', fontsize=20)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) 
    plt.savefig(output_path)
    plt.close()
    print(f"  -> [成功] 平滑化検証グラフを保存しました: {os.path.basename(output_path)}")


def analyze_video_file(video_path, output_dir, analysis_roi, bin_thresh, thick_thresh, lookahead_window, min_in_window, smoothing_window, create_preview, create_graph):
    """(v13.6 と変更なし)"""
    
    print(f"  -> v10.5ロジック (Hybrid Mask) で全編解析を開始...")
    base = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = os.path.join(output_dir, f"{base}_tips_coordinates.csv")
    x_roi, y_roi, w_roi, h_roi = analysis_roi
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [エラー] 動画 {video_path} が開けません。スキップします。")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_number = 0
    results_list = [] 
    kernel_3x3 = np.ones((3,3), np.uint8)
    all_thickness_values = [] 
    bin_closed_video_path = os.path.join(output_dir, f"{base}_binary_closed.avi")
    out_bin_closed = cv2.VideoWriter(bin_closed_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w_roi, h_roi), isColor=False)

    while True:
        ret, frame = cap.read()
        if not ret: break
        cat_tip_global = None
        gw_tip_global = None
        frame_roi = frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
        if bin_thresh == 0:
            _, binary_all = cv2.threshold(cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, binary_all = cv2.threshold(cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY), bin_thresh, 255, cv2.THRESH_BINARY_INV)
        device_mask = get_main_body_mask(binary_all)
        closed_roi = cv2.morphologyEx(binary_all, cv2.MORPH_CLOSE, kernel_3x3, iterations=CLOSING_ITERATIONS)
        gw_safe_mask = get_main_body_mask(closed_roi)
        out_bin_closed.write(gw_safe_mask) 
        gw_skeleton = cv2.ximgproc.thinning(gw_safe_mask)
        gw_endpoints = find_endpoints(gw_skeleton)
        gw_tip_roi = get_endpoint_by_x(gw_endpoints, find_min=True)
        cat_tip_roi = None
        if gw_tip_roi:
            dist_map = cv2.distanceTransform(device_mask, cv2.DIST_L2, 5)
            current_thickness_values = dist_map[dist_map > 0]
            if current_thickness_values.size > 0:
                all_thickness_values.extend(current_thickness_values)
            skeleton = cv2.ximgproc.thinning(device_mask)
            trace_data = trace_skeleton_and_get_thickness_and_coords(skeleton, dist_map, gw_tip_roi)
            if trace_data:
                cat_tip_roi, _ = find_junction_with_lookahead(
                    trace_data, thick_thresh, lookahead_window, min_in_window
                )
        cat_tip_global = (cat_tip_roi[0] + x_roi, cat_tip_roi[1] + y_roi) if cat_tip_roi else None
        gw_tip_global = (gw_tip_roi[0] + x_roi, gw_tip_roi[1] + y_roi) if gw_tip_roi else None
        results_list.append({
            'frame': frame_number,
            'catheter_tip_x': cat_tip_global[0] if cat_tip_global else np.nan,
            'catheter_tip_y': cat_tip_global[1] if cat_tip_global else np.nan,
            'guidewire_tip_x': gw_tip_global[0] if gw_tip_global else np.nan,
            'guidewire_tip_y': gw_tip_global[1] if gw_tip_global else np.nan
        })
        frame_number += 1
    cap.release()
    out_bin_closed.release()
    print(f"  -> [成功] デバッグ用二値化動画を保存: {os.path.basename(bin_closed_video_path)}")
    if not results_list:
        print("  -> [エラー] 動画からフレームが読み取れなかったか、空の動画です。")
        return
    print(f"  -> 全 {frame_number} フレームの解析完了。")
    print("  -> 平滑化処理を実行中...")
    df_results = pd.DataFrame(results_list)
    df_results['time_s'] = df_results['frame'] / fps
    df_results['catheter_tip_x'] = df_results['catheter_tip_x'].fillna(method='ffill')
    df_results['catheter_tip_y'] = df_results['catheter_tip_y'].fillna(method='ffill')
    df_results['guidewire_tip_x'] = df_results['guidewire_tip_x'].fillna(method='ffill')
    df_results['guidewire_tip_y'] = df_results['guidewire_tip_y'].fillna(method='ffill')
    df_results['smooth_cath_x'] = df_results['catheter_tip_x'].rolling(window=smoothing_window, center=True, min_periods=1).mean()
    df_results['smooth_cath_y'] = df_results['catheter_tip_y'].rolling(window=smoothing_window, center=True, min_periods=1).mean()
    df_results['smooth_gw_x'] = df_results['guidewire_tip_x'].rolling(window=smoothing_window, center=True, min_periods=1).mean()
    df_results['smooth_gw_y'] = df_results['guidewire_tip_y'].rolling(window=smoothing_window, center=True, min_periods=1).mean()
    df_results.to_csv(csv_path, index=False)
    print(f"  -> [成功] 平滑化済み座標CSVを保存しました: {os.path.basename(csv_path)}")
    if all_thickness_values:
        hist_path = os.path.join(output_dir, f"{base}_thickness_histogram.png")
        try:
            plt.figure(figsize=(12, 7))
            plt.hist(all_thickness_values, bins=100, range=(0, 5), label="Thickness Distribution")
            plt.axvline(THICKNESS_THRESHOLD, color='red', linestyle='--', 
                        label=f'Threshold ({THICKNESS_THRESHOLD})')
            plt.title(f"Thickness Histogram (All Frames) - {base}", fontsize=20)
            plt.xlabel("Thickness (Distance Transform Value)", fontsize=16)
            plt.ylabel("Frequency (Pixel Count)", fontsize=16)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend(fontsize=14)
            plt.yscale('log') 
            plt.tight_layout()
            plt.savefig(hist_path)
            plt.close()
            print(f"  -> [成功] 太さヒストグラムを保存しました: {os.path.basename(hist_path)}")
        except Exception as e:
            print(f"  [!!! エラー !!!] 太さヒストグラムの作成に失敗しました: {e}")
    else:
        print(f"  -> [警告] 太さ情報が収集できなかったため、ヒストグラムをスキップします。")
    if create_graph:
        graph_path = os.path.join(output_dir, f"{base}_smoothing_verification.png")
        try:
            plot_smoothing_comparison(df_results, graph_path)
        except Exception as e:
            print(f"  [!!! エラー !!!] 検証グラフの作成に失敗しました: {e}")
    if create_preview:
        print(f"  -> 平滑化後のプレビュー動画を作成中...")
        video_out_path = os.path.join(output_dir, f"{base}_tips_preview_SMOOTHED.avi")
        try:
            cap = cv2.VideoCapture(video_path) 
            out_vid = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (vid_w, vid_h))
            for frame_idx, row in df_results.iterrows():
                ret, frame = cap.read()
                if not ret: break
                output_frame = frame.copy()
                smooth_cath_x = row['smooth_cath_x']
                smooth_cath_y = row['smooth_cath_y']
                smooth_gw_x = row['smooth_gw_x']
                smooth_gw_y = row['smooth_gw_y']
                cv2.rectangle(output_frame, (x_roi, y_roi), (x_roi + w_roi, y_roi + h_roi), (0, 255, 0), 1)
                if not pd.isna(smooth_cath_x):
                    cv2.circle(output_frame, (int(smooth_cath_x), int(smooth_cath_y)), 6, (0, 255, 255), 2) 
                if not pd.isna(smooth_gw_x):
                    cv2.circle(output_frame, (int(smooth_gw_x), int(smooth_gw_y)), 6, (255, 255, 0), 2) 
                out_vid.write(output_frame)
            cap.release()
            out_vid.release()
            print(f"  -> [成功] 平滑化プレビュー動画を保存しました: {os.path.basename(video_out_path)}")
        except Exception as e:
            print(f"  [!!! エラー !!!] プレビュー動画の作成に失敗しました: {e}")


# --- ★★★ メイン実行ブロック (v13.8 Interactive Selection) ★★★ ---
if __name__ == "__main__":
    
    print("--- ワークフロー開始 (v13.8 Interactive Selection) ---")
    
    input_folder = INPUT_DIR
    output_folder = OUTPUT_DIR
    
    if not os.path.isdir(input_folder):
        print(f"[エラー] 入力フォルダが見つかりません: {input_folder}")
        exit()
        
    os.makedirs(output_folder, exist_ok=True)
    print(f"入力フォルダ: {input_folder}")
    print(f"出力フォルダ: {output_folder}")

    glob_pattern = os.path.join(input_folder, f"*{VIDEO_EXTENSION}")
    video_files = sorted(glob.glob(glob_pattern))

    if not video_files:
        print(f"[エラー] 入力フォルダ内に '{VIDEO_EXTENSION}' の動画が見つかりませんでした。")
        exit()

    num_videos = len(video_files)
    print(f"\n--- {num_videos} 件の動画が検出されました ---")
    print("各動画について、個別に再解析を実行するか選択してください。")

    # --- 1. 全動画をループ ---
    for i, video_path in enumerate(video_files):
        print("\n" + "="*70)
        print(f"動画 ({i+1}/{num_videos}): {os.path.basename(video_path)}")
        
        # --- 2. ユーザーに実行可否を尋ねる ---
        choice = input(f" -> この動画を個別にROI指定して再解析しますか？ (Yes: y / No: n / 終了: q) [Y/n/q]: ").strip().lower()
        
        if choice == 'q':
            print("--- 処理を中断します ---")
            break
        elif choice == 'n':
            print("  -> スキップします。")
            continue
        elif choice == 'y' or choice == '':
            # --- 3. (Yes の場合) ROI選択と解析を実行 ---
            print(f"--- 個別処理を開始: {os.path.basename(video_path)} ---")
            try:
                # 3a. ROIを個別に選択
                analysis_roi = select_analysis_roi(video_path)
                
                if analysis_roi:
                    # 3b. ROIが選択されたら、解析を実行
                    print(f"--- 選択されたROI {analysis_roi} で再解析を実行します ---")
                    analyze_video_file(
                        video_path, 
                        output_folder, 
                        analysis_roi, 
                        BIN_THRESHOLD, 
                        THICKNESS_THRESHOLD, 
                        LOOKAHEAD_WINDOW, 
                        MIN_IN_WINDOW,
                        MOVING_AVERAGE_WINDOW,
                        CREATE_PREVIEW_VIDEO,
                        CREATE_VERIFICATION_GRAPH 
                    )
                    print(f"--- {os.path.basename(video_path)} の処理が完了しました ---")
                else:
                    print("  -> ROI選択がキャンセルされたため、この動画をスキップします。")

            except Exception as e:
                print(f"\n[!!! 重大なエラー !!!] {os.path.basename(video_path)} の処理中にエラーが発生しました: {e}")
        else:
            print(f"  -> '{choice}' は無効な入力です。スキップします。")


    print("\n" + "="*70)
    print("--- 全ての動画の確認が完了しました ---")