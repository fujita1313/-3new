import cv2
import numpy as np
import os
import pandas as pd
from collections import deque
import time 
import matplotlib.pyplot as plt


# --- ★★★ ユーザー設定 (v13.7 Debug Subplots) ★★★ ---

# 1. 解析対象の「動画ファイル」
VIDEO_PATH = r"C:\Users\haruk\OneDrive\ドキュメント\修士\実験データ\20251022\解析結果\時間同期\カテーテル\trimmed_videos\IMG_0627_trimmed.avi"

# 2. 解析結果の「出力ファイル名」
OUTPUT_GRAPH_NAME = "lookahead_validation_subplots_IMG_0627.png"
OUTPUT_IMAGE_NAME = "validation_image_IMG_0627.png"

# 3. v10.5 アルゴリズム設定
BIN_THRESHOLD = 0 
THICKNESS_THRESHOLD = 1.3
LOOKAHEAD_WINDOW = 25
MIN_IN_WINDOW = 13 
CLOSING_ITERATIONS = 1

# --- ユーティリティ関数群 (v13.6 と変更なし) ---

def select_analysis_roi(video_path):
    """ROIを1つ選択させる"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        print(f"  [エラー] 動画 {video_path} が開けません。")
        return None, None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    target_frame_idx = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret: 
        print(f"  [エラー] ROI選択用のフレーム読み込み失敗 (Frame: {target_frame_idx})")
        return None, None
        
    print("  --- ROIの選択 ---")
    print("  解析したい領域をドラッグして囲み、[Enter]キーを押してください。")
    h, w, _ = frame.shape
    scale = min(1.0, 1200 / w)
    preview_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    
    window_title = f"Select Analysis Area for {os.path.basename(video_path)} (Press Enter)"
    roi_preview = cv2.selectROI(window_title, preview_frame, False)
    cv2.destroyWindow(window_title)
    
    roi = tuple(int(c / scale) for c in roi_preview)
    
    if roi[2] == 0 or roi[3] == 0:
        print("  -> ROI選択がキャンセルされました。")
        return None, None
        
    print(f"  -> ROI {roi} が選択されました。")
    return frame, roi

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
    
    if not (0 <= start_point[1] < h and 0 <= start_point[0] < w):
        print(f"  [エラー] 開始点 {start_point} が骨格の範囲外です。")
        return []
        
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

# --- ★★★ メイン実行ブロック (グラフB + 可視化画像) ★★★ ---
def create_debug_outputs():
    
    print(f"--- デバッグ (v13.7 Subplots) スクリプト開始 ---")
    print(f"解析対象: {VIDEO_PATH}")
    
    if not os.path.exists(VIDEO_PATH):
        print(f"[エラー] 動画ファイルが見つかりません。パスを確認してください。")
        return

    # --- 1. 代表フレームとROIの取得 ---
    representative_frame, analysis_roi = select_analysis_roi(VIDEO_PATH)
    
    if representative_frame is None or analysis_roi is None:
        print("[エラー] フレームまたはROIが取得できませんでした。処理を終了します。")
        return

    x_roi, y_roi, w_roi, h_roi = analysis_roi
    frame_roi = representative_frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
    visualization_image = frame_roi.copy()

    # --- 2. 画像処理 ---
    print("  -> 画像処理を実行中...")
    if BIN_THRESHOLD == 0:
        _, binary_all = cv2.threshold(cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, binary_all = cv2.threshold(cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY), BIN_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    kernel_3x3 = np.ones((3,3), np.uint8)
    device_mask = get_main_body_mask(binary_all)
    closed_roi = cv2.morphologyEx(binary_all, cv2.MORPH_CLOSE, kernel_3x3, iterations=CLOSING_ITERATIONS)
    gw_safe_mask = get_main_body_mask(closed_roi)

    # --- 3. GW先端の特定 ---
    gw_skeleton = cv2.ximgproc.thinning(gw_safe_mask)
    gw_endpoints = find_endpoints(gw_skeleton)
    gw_tip_roi = get_endpoint_by_x(gw_endpoints, find_min=True) 

    if gw_tip_roi is None:
        print("[エラー] GW先端が見つかりませんでした。")
        return
    print(f"  -> GW先端を特定: {gw_tip_roi}")
    cv2.circle(visualization_image, gw_tip_roi, 8, (255, 255, 0), 2) # Cyan

    # --- 4. トレースデータの取得 ---
    dist_map = cv2.distanceTransform(device_mask, cv2.DIST_L2, 5)
    skeleton = cv2.ximgproc.thinning(device_mask)
    print("  -> 骨格トレースと太さデータ収集を実行中...")
    trace_data = trace_skeleton_and_get_thickness_and_coords(skeleton, dist_map, gw_tip_roi)

    if not trace_data:
        print("[エラー] 骨格のトレースに失敗しました。")
        return
    print(f"  -> トレース完了。{len(trace_data)} 点のデータを取得。")

    # --- 5. カテーテル先端の特定 ---
    print("  -> カテーテル先端を特定中 (Lookahead)...")
    cat_tip_roi, cat_dist = find_junction_with_lookahead(
        trace_data, THICKNESS_THRESHOLD, LOOKAHEAD_WINDOW, MIN_IN_WINDOW
    )
    
    if cat_tip_roi:
        print(f"  -> カテーテル先端を特定: {cat_tip_roi} (距離: {cat_dist:.2f})")
        cv2.circle(visualization_image, cat_tip_roi, 8, (0, 255, 255), 2) # Yellow
    else:
        print("  -> [警告] カテーテル先端は検出されませんでした。")

    # --- 6. 可視化画像の保存 ---
    try:
        cv2.imwrite(OUTPUT_IMAGE_NAME, visualization_image)
        print(f"[成功] 可視化画像を '{OUTPUT_IMAGE_NAME}' として保存しました。")
    except Exception as e:
        print(f"[エラー] 可視化画像の保存に失敗しました: {e}")

    # --- 7. 検証用データの計算 ---
    print(f"  -> グラフ用データを計算中...")
    n_points = len(trace_data)
    distances = [item[0] for item in trace_data]
    raw_thickness_values = np.array([item[1] for item in trace_data])
    lookahead_results = [] 
    
    for i in range(n_points):
        start_index = i
        end_index = min(i + LOOKAHEAD_WINDOW, n_points)
        window_data = raw_thickness_values[start_index:end_index]
        count_above_thresh = np.sum(window_data > THICKNESS_THRESHOLD)
        lookahead_results.append(count_above_thresh)

    # --- 8. ★★★ 上下2段グラフ (Subplots) の生成 ★★★ ---
    print(f"  -> 上下2段グラフを生成中...")
    
    # 2つのサブプロットを作成 (X軸を共有)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    # --- 上段 (グラフA: 生データ) ---
    ax1.plot(distances, raw_thickness_values, label='Raw Thickness', color='green', 
             marker='o', linestyle='None', markersize=4,alpha=0.5)

    ax1.axhline(y=THICKNESS_THRESHOLD, color='red', linestyle='--', 
                label=f'Thickness Threshold ({THICKNESS_THRESHOLD})')
    if cat_dist is not None:
        ax1.axvline(x=cat_dist, color='yellow', linestyle='-', linewidth=3,
                    label=f'Detected Junction (Dist={cat_dist:.2f})')
    
    ax1.set_ylabel('Raw Thickness (pixels)', fontsize=28)
    ax1.tick_params(axis='y', labelsize=24)
    ax1.legend(fontsize=20, loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.8)
    # 上段のY軸の範囲を調整 (0から3.0まで)
    ax1.set_ylim(0, 3.0)

    # --- 下段 (グラフB: 検証スコア) ---
    ax2.plot(distances, lookahead_results, label=f'Lookahead Count (W={LOOKAHEAD_WINDOW})', color='blue')
    ax2.axhline(y=MIN_IN_WINDOW, color='red', linestyle='--', 
                label=f'Min Count Threshold (N_min={MIN_IN_WINDOW})')
    if cat_dist is not None:
        ax2.axvline(x=cat_dist, color='yellow', linestyle='-', linewidth=3,
                    label=f'Detected Junction (Dist={cat_dist:.2f})')

    # 共通のX軸ラベル
    ax2.set_xlabel('Distance from GW tip (pixels)', fontsize=28)
    # Y軸ラベル
    y_label = f'Count above {THICKNESS_THRESHOLD} in next {LOOKAHEAD_WINDOW} px'
    ax2.set_ylabel(y_label, fontsize=28)
    
    ax2.tick_params(axis='x', labelsize=24)
    ax2.tick_params(axis='y', labelsize=24)
    ax2.legend(fontsize=20, loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.8)
    # 下段のY軸の範囲を調整 (0からウィンドウサイズ+1まで)
    ax2.set_ylim(0, LOOKAHEAD_WINDOW + 1)

    # 全体のレイアウトを調整
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # 共通タイトルがないため、上マージンを詰める
    
    # --- 9. グラフ保存 ---
    try:
        plt.savefig(OUTPUT_GRAPH_NAME)
        print(f"[成功] 検証グラフを '{OUTPUT_GRAPH_NAME}' として保存しました。")
    except Exception as e:
        print(f"[エラー] グラフの保存に失敗しました: {e}")
    plt.close()

    print("\n" + "="*70)
    print("--- 全てのデバッグ出力が完了しました ---")
    print("="*70)

# --- スクリプト実行 ---
if __name__ == "__main__":
    create_debug_outputs()