import cv2
import numpy as np
import os
import pandas as pd
import glob
from collections import deque
import time 
import matplotlib.pyplot as plt

# --- ★★★ ユーザー設定 ★★★ ---
VIDEO_PATH = r"C:\Users\haruk\OneDrive\ドキュメント\修士\実験データ\20251022\解析結果\時間同期\カテーテル\trimmed_videos\IMG_0645_trimmed.avi"
OUTPUT_DIR = r"C:\Users\haruk\OneDrive\ドキュメント\修士\実験データ\20251022\解析結果\v12_Full_RawLookahead"
BIN_THRESHOLD = 0 
THICKNESS_THRESHOLD = 1.3
LOOKAHEAD_WINDOW = 25
MIN_IN_WINDOW = 13 
# --- 設定ここまで ---



# --- ユーティリティ関数群 (変更なし) ---

def select_analysis_roi(video_path):
    """ROIを1つ選択させる"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    target_frame = int(5.0 * fps) 
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()
    if not ret: 
        print("ROI選択用のフレーム読み込み失敗")
        return None
    print("--- ROIの選択 ---")
    print("解析したい領域をドラッグして囲み、[Enter]キーを押してください。")
    h, w, _ = frame.shape
    scale = min(1.0, 1200 / w)
    preview_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    roi_preview = cv2.selectROI("Select Analysis Area (Press Enter)", preview_frame, False)
    cv2.destroyWindow("Select Analysis Area (Press Enter)")
    roi = tuple(int(c / scale) for c in roi_preview)
    if roi[2] == 0 or roi[3] == 0:
        return None
    return roi

def get_main_body_mask(mask):
    """二値化マスクから最大の連結成分(塊)だけを抽出"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask)
    main_contour = max(contours, key=cv2.contourArea)
    main_body_mask = np.zeros_like(mask)
    cv2.drawContours(main_body_mask, [main_contour], -1, 255, thickness=cv2.FILLED)
    return main_body_mask

def find_endpoints(thinned_image):
    """細線化された画像の端点を見つける"""
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    thinned_01 = thinned_image // 255
    neighbor_count = cv2.filter2D(thinned_01, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    endpoint_mask = ((thinned_01 == 1) & (neighbor_count == 1))
    rows, cols = np.where(endpoint_mask)
    return list(zip(cols, rows)) # (x, y)

def get_endpoint_by_x(endpoints, find_min=True):
    """端点リストからX座標が最小(または最大)の点を見つける"""
    if not endpoints:
        return None
    if find_min:
        return min(endpoints, key=lambda p: p[0])
    else:
        return max(endpoints, key=lambda p: p[0])

def trace_skeleton_and_get_thickness_and_coords(skeleton, dist_map, start_point):
    """「[距離, 太さ, x, y]」のリストを返す"""
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
    """
    「生」の太さリストを探索し、「先読み」ロジックで連結部を見つける
    ★ 戻り値を ( (x, y), 距離 ) のタプルに変更
    """
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
            junction_dist = trace_data[start_index][0] # ★ 距離も取得
            return junction_point, junction_dist # (x, y) と 距離 を返す
            
    return None, None

# --- フェーズ1: 全編解析 (v10.5ロジック) ---
def run_full_analysis(video_path, output_dir, analysis_roi, bin_thresh, thick_thresh, lookahead_window, min_in_window):
    print("--- フェーズ1: 全編解析を実行中 ---")
    os.makedirs(output_dir, exist_ok=True)
    
    base = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = os.path.join(output_dir, f"{base}_tips_coordinates.csv")
    video_out_path = os.path.join(output_dir, f"{base}_tips_preview.avi")
    dist_map_path = os.path.join(output_dir, f"{base}_dist_map_preview.avi") 
    
    x_roi, y_roi, w_roi, h_roi = analysis_roi

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out_vid = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (vid_w, vid_h))
    out_dist = cv2.VideoWriter(dist_map_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w_roi, h_roi))
    
    frame_number = 0
    with open(csv_path, 'w', newline='') as f:
        f.write("frame,catheter_tip_x,catheter_tip_y,guidewire_tip_x,guidewire_tip_y\n")
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_roi = frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
            
            # --- v10.5 ロジック ---
            if bin_thresh == 0:
                _, binary_all = cv2.threshold(cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            else:
                _, binary_all = cv2.threshold(cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY), bin_thresh, 255, cv2.THRESH_BINARY_INV)

            device_mask = get_main_body_mask(binary_all)
            dist_map = cv2.distanceTransform(device_mask, cv2.DIST_L2, 5)
            skeleton = cv2.ximgproc.thinning(device_mask)
            endpoints = find_endpoints(skeleton)
            
            cat_tip_roi = None
            gw_tip_roi = get_endpoint_by_x(endpoints, find_min=True)
            
            if gw_tip_roi:
                trace_data = trace_skeleton_and_get_thickness_and_coords(skeleton, dist_map, gw_tip_roi)
                if trace_data:
                    cat_tip_roi, _ = find_junction_with_lookahead( # ★ 距離は使わない
                        trace_data, 
                        thick_thresh, 
                        lookahead_window, 
                        min_in_window
                    )

            cat_tip_global = (cat_tip_roi[0] + x_roi, cat_tip_roi[1] + y_roi) if cat_tip_roi else None
            gw_tip_global = (gw_tip_roi[0] + x_roi, gw_tip_roi[1] + y_roi) if gw_tip_roi else None
                
            f.write(f"{frame_number},{cat_tip_global[0] if cat_tip_global else 'NaN'},{cat_tip_global[1] if cat_tip_global else 'NaN'},{gw_tip_global[0] if gw_tip_global else 'NaN'},{gw_tip_global[1] if gw_tip_global else 'NaN'}\n")

            # --- 可視化 ---
            output_frame = frame.copy()
            cv2.rectangle(output_frame, (x_roi, y_roi), (x_roi + w_roi, y_roi + h_roi), (0, 255, 0), 1)
            dist_map_vis = cv2.normalize(dist_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            dist_map_vis_color = cv2.applyColorMap(dist_map_vis, cv2.COLORMAP_JET)
            dist_map_vis_color[device_mask == 0] = (0, 0, 0) 
            out_dist.write(dist_map_vis_color) 
            output_frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi][skeleton == 255] = (255, 0, 255)
            if cat_tip_global: cv2.circle(output_frame, cat_tip_global, 6, (0, 255, 255), 2) 
            if gw_tip_global: cv2.circle(output_frame, gw_tip_global, 6, (255, 255, 0), 2) 
            out_vid.write(output_frame)
            frame_number += 1
            
    cap.release()
    out_vid.release()
    out_dist.release() 
    print(f"--- フェーズ1 完了 ---")
    print(f"  座標CSV: {csv_path}")
    print(f"  プレビュー動画: {video_out_path}")
    return video_out_path

# --- フェーズ2: 問題フレームの特定 (v11.1) ---
def find_problematic_frames(preview_video_path):
    print(f"\n--- フェーズ2: 問題フレームの特定 ---")
    print(f"  解析済み動画: {preview_video_path} を開きます...")
    
    cap = cv2.VideoCapture(preview_video_path)
    if not cap.isOpened():
        print(f"エラー: 解析済み動画 {preview_video_path} を開けませんでした。")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0 
        
    current_frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
    pause = True
    
    selected_frame_indices = []
    
    print("\n--- フレーム特定ツール ---")
    print(f"  [Space]: 再生 / 一時停止")
    print(f"  [->] (n): 1フレーム進む")
    print(f"  [<-] (p): 1フレーム戻る")
    print(f"  [s]:     現在のフレーム番号をコンソールに記録")
    print(f"  [q]:     終了")
    print("-------------------------")

    window_name = "Frame Finder (Space=Play/Pause, s=Bookmark)"

    while True:
        if not pause:
            ret, frame = cap.read()
            if ret:
                current_frame_idx += 1
            else:
                print("動画の終わりに達しました。")
                pause = True
                current_frame_idx = total_frames - 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
                ret, frame = cap.read()
                if not ret: break 
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"エラー: フレーム {current_frame_idx} の読み込みに失敗しました。")
                break
        
        display_frame = frame.copy()
        time_s = current_frame_idx / fps
        cv2.putText(display_frame, f"Frame: {current_frame_idx} / {total_frames} ({time_s:.2f}s)", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if pause:
            cv2.putText(display_frame, "PAUSED", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow(window_name, display_frame)

        wait_time = int(1000 / fps) if not pause else 0
        key = cv2.waitKey(wait_time) & 0xFF

        if key == ord('q'): 
            break
        elif key == ord(' '): 
            pause = not pause
        elif key == ord('s'): 
            if current_frame_idx not in selected_frame_indices:
                selected_frame_indices.append(current_frame_idx)
                print(f"  *** フレーム {current_frame_idx} (@ {time_s:.2f}s) を記録しました (計{len(selected_frame_indices)}件) ***")
            else:
                print(f"  (フレーム {current_frame_idx} は既に記録済みです)")
        
        elif key == ord('n') or key == 83: 
            pause = True 
            current_frame_idx = min(current_frame_idx + 1, total_frames - 1)
        elif key == ord('p') or key == 81: 
            pause = True
            current_frame_idx = max(current_frame_idx - 1, 0)
            
    cap.release()
    cv2.destroyAllWindows()
    print(f"--- フェーズ2 完了 ---")
    return sorted(selected_frame_indices) 

# --- フェーズ3: 問題フレームのグラフ描画 (v10.5ロジック) ---
def plot_analysis_for_frames(original_video_path, analysis_roi, frame_indices_to_plot, bin_thresh, thick_thresh, lookahead_window, min_in_window):
    print(f"\n--- フェーズ3: 問題フレームのグラフ描画 ---")
    print(f"  対象フレーム: {frame_indices_to_plot}")
    
    cap = cv2.VideoCapture(original_video_path)
    if not cap.isOpened():
        print(f"エラー: 元動画 {original_video_path} が開けません。")
        return
        
    x_roi, y_roi, w_roi, h_roi = analysis_roi

    num_plots = len(frame_indices_to_plot)
    fig, axes = plt.subplots(num_plots, 2, figsize=(20, 7 * num_plots), squeeze=False) 
    
    max_dist_overall = 0 
    plot_data_list = [] # (x_axis, y_raw, frame_idx, visual_frame, junction_dist) を格納

    for i, frame_idx in enumerate(frame_indices_to_plot):
        print(f"\n  ステップ {i+1}/{num_plots}: フレーム {frame_idx} を再解析中...")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"  -> エラー: フレーム {frame_idx} が読み込めません。")
            continue
            
        frame_roi = frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
        
        # --- v10.5 ロジック ---
        if bin_thresh == 0:
            _, binary_all = cv2.threshold(cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, binary_all = cv2.threshold(cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY), bin_thresh, 255, cv2.THRESH_BINARY_INV)

        device_mask = get_main_body_mask(binary_all)
        dist_map = cv2.distanceTransform(device_mask, cv2.DIST_L2, 5)
        skeleton = cv2.ximgproc.thinning(device_mask)
        endpoints = find_endpoints(skeleton)
        gw_tip_roi = get_endpoint_by_x(endpoints, find_min=True)
        
        cat_tip_roi = None
        junction_dist = None
        x_axis_distance = None
        y_axis_thickness = None
        
        if gw_tip_roi is None:
            print(f"  -> エラー: フレーム {frame_idx} でGW先端が見つかりません。")
        else:
            trace_data = trace_skeleton_and_get_thickness_and_coords(skeleton, dist_map, gw_tip_roi)
            if not trace_data:
                print(f"  -> エラー: フレーム {frame_idx} でトレースに失敗しました。")
            else:
                x_axis_distance = [item[0] for item in trace_data]
                y_axis_thickness = [item[1] for item in trace_data]
                
                # ★ 先読みロジックで連結部を検出
                cat_tip_roi, junction_dist = find_junction_with_lookahead(
                    trace_data, 
                    thick_thresh, 
                    lookahead_window, 
                    min_in_window
                )
                
                if np.max(x_axis_distance) > max_dist_overall:
                    max_dist_overall = np.max(x_axis_distance)

        # --- 可視化フレーム作成 ---
        output_frame_roi = frame_roi.copy()
        output_frame_roi[skeleton == 255] = (255, 0, 255) 
        if cat_tip_roi:
            cv2.circle(output_frame_roi, cat_tip_roi, 6, (0, 255, 255), 2) 
        if gw_tip_roi:
            cv2.circle(output_frame_roi, gw_tip_roi, 6, (255, 255, 0), 2) 
        
        plot_data_list.append((x_axis_distance, y_axis_thickness, frame_idx, output_frame_roi, junction_dist))

    cap.release()
    print("\n--- 全フレーム解析完了。グラフを描画します ---")

    # --- グラフの描画 (ループ) ---
    for i, (x_axis, y_raw, frame_idx, visual_frame, junction_dist) in enumerate(plot_data_list):
        
        # --- 左側のプロット (画像) ---
        ax_img = axes[i, 0] 
        ax_img.imshow(cv2.cvtColor(visual_frame, cv2.COLOR_BGR2RGB))
        ax_img.set_title(f"Visual Result (Frame {frame_idx})")
        ax_img.axis('off')
        
        # --- 右側のプロット (グラフ) ---
        ax_graph = axes[i, 1] 
        if x_axis: 
            ax_graph.plot(x_axis, y_raw, marker='.', markersize=2, linestyle='-', label='Original Thickness', alpha=0.5)
            
            # 水平の閾値線
            ax_graph.axhline(thick_thresh, color='cyan', linestyle='--', label=f'Threshold ({thick_thresh})')
            
            # ★★★ 検出した連結部の「垂直線」を追加 ★★★
            if junction_dist is not None:
                ax_graph.axvline(junction_dist, color='yellow', linestyle='--', linewidth=2, label=f'Detected Junction (Dist={junction_dist:.1f})')
            
            ax_graph.set_title(f"Raw Thickness along Skeleton Path @ Frame {frame_idx}")
            ax_graph.set_ylabel("Thickness (Distance Transform Value)")
            ax_graph.grid(True, which="both", linestyle='--', alpha=0.5)
            ax_graph.set_xlim(0, max_dist_overall + 50) 
            ax_graph.set_xticks(np.arange(0, max_dist_overall + 50, 50))
            ax_graph.legend()
        else:
             ax_graph.set_title(f"Analysis Failed (Frame {frame_idx})")

    axes[-1, 1].set_xlabel("Distance from GW Tip (pixels)")
    plt.tight_layout() 
    plt.show()


# --- メイン実行ブロック ---
if __name__ == "__main__":
    
    # --- 1. 共通ROIの選択 (1回だけ) ---
    print("--- ワークフロー開始 ---")
    print("--- フェーズ1/4: 共通ROIの選択 ---")
    analysis_roi = select_analysis_roi(VIDEO_PATH)
    
    if analysis_roi:
        print(f"ROI {analysis_roi} で解析を開始します。")
        
        # --- 2. 全編解析 (v10.5) ★★★ 変更点 ★★★ ---
        print("\n" + "="*50)
        print(f"--- フェーズ2/4: 全編解析 (v10.5 - Raw + Lookahead) (T={THICKNESS_THRESHOLD}, L={LOOKAHEAD_WINDOW}, M={MIN_IN_WINDOW}) ---")
        preview_path = run_full_analysis(
            VIDEO_PATH, 
            OUTPUT_DIR, 
            analysis_roi, 
            BIN_THRESHOLD, 
            THICKNESS_THRESHOLD, 
            LOOKAHEAD_WINDOW, # ★ 引数を変更
            MIN_IN_WINDOW     # ★ 引数を変更
        )
        
        if preview_path and os.path.exists(preview_path):
            # --- 3. 問題フレームの特定 (v11.1) ---
            print("\n" + "="*50)
            print("--- フェーズ3/4: 問題フレームの特定 (v11.1) ---")
            problem_frames = find_problematic_frames(preview_path)
            
            if problem_frames:
                # --- 4. グラフ描画 (v10.5) ★★★ 変更点 ★★★ ---
                print("\n" + "="*50)
                print("--- フェーズ4/4: 問題フレームのグラフ表示 (v10.5 - Raw + Lookahead) ---")
                plot_analysis_for_frames(
                    VIDEO_PATH, 
                    analysis_roi, 
                    problem_frames, 
                    BIN_THRESHOLD, 
                    THICKNESS_THRESHOLD, 
                    LOOKAHEAD_WINDOW, # ★ 引数を変更
                    MIN_IN_WINDOW     # ★ 引数を変更
                )
            else:
                print("問題フレームが選択されなかったため、グラフ表示をスキップします。")
        else:
            print("全編解析のプレビュー動画が見つからなかったため、処理を中断します。")
            
    print("\n--- 全ワークフロー完了 ---")