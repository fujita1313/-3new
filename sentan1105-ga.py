import cv2
import os
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt 

# --- ★★★ ユーザー設定 ★★★ ---
INPUT_DIR = r"C:\Users\haruk\OneDrive\ドキュメント\修士\実験データ\20251022\解析結果\時間同期\ガイドワイヤ\trimmed_videos" 
OUTPUT_DIR = r"C:\Users\haruk\OneDrive\ドキュメント\修士\実験データ\20251022\解析結果\ガイドワイヤ\先端検出結果_v23_CollisionZone"
KNOWN_LENGTH_MM = 100.0
MANUAL_THRESHOLD = 130
COLLISION_PIXEL_THRESHOLD = 10 
CLOSING_ITERATIONS = 2
ROI_FRAME_TIME_S = 5.0 
MAX_DISPLAY_HEIGHT = 800 


# --- ★★★ 修正箇所 (v23.6) ★★★ ---
def select_polygon_roi(video_path, window_title="STEP 1: Draw Polygon ROI (Tube Interior)"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    target_frame = int(ROI_FRAME_TIME_S * fps) 
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()
    if not ret: 
        print("ROI選択用のフレーム読み込み失敗")
        return None, None

    print(f"\n--- ポリゴンROIの選択 ({window_title}) ---")
    print(" 1. マウスで頂点を順にクリックしてください。")
    print(" 2. [u] キーで、直前のクリックを取り消せます。")
    print(" 3. [Enter] キーで、ポリゴンを閉じて決定します。")
    
    h, w, _ = frame.shape
    scale = min(1.0, MAX_DISPLAY_HEIGHT / h)
    preview_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    
    points_preview = [] 
    
    # ★ 修正: window_name -> window_title
    cv2.namedWindow(window_title)
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points_preview.append((x, y))

    # ★ 修正: window_name -> window_title
    cv2.setMouseCallback(window_title, mouse_callback)

    while True:
        draw_frame = preview_frame.copy()
        if len(points_preview) > 0:
            cv2.polylines(draw_frame, [np.array(points_preview)], isClosed=False, color=(0, 255, 0), thickness=2)
            for p in points_preview:
                cv2.circle(draw_frame, p, 4, (0, 0, 255), -1)
        
        # ★ 修正: window_name -> window_title
        cv2.imshow(window_title, draw_frame)
        key = cv2.waitKey(20) & 0xFF
        
        if key == 13: # Enter
            if len(points_preview) >= 3:
                print("ポリゴンを決定しました。")
                break
            else:
                print("エラー: 最低3つの頂点が必要です。")
        elif key == ord('u'): # Undo
            if points_preview: points_preview.pop()
        elif key == ord('q'): # Quit
            cv2.destroyAllWindows()
            return None, None
            
    cv2.destroyAllWindows()
    
    points_original = [ (int(p[0] / scale), int(p[1] / scale)) for p in points_preview ]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points_original)], 255)
    
    return mask, (h, w) # マスクと、元のフレームサイズを返す
# --- ★★★ 修正ここまで ★★★ ---


# --- (select_polygon_collision_zone 関数は v23.4 と同じ) ---
def select_polygon_collision_zone(video_path):
    """
    衝突判定を行う「閉塞部」のポリゴンROIを選択させる
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 2)
    ret, frame = cap.read()
    cap.release()
    if not ret: 
        print("最終フレームの読み込み失敗")
        return None

    print("\n--- 衝突ゾーンROIの選択 (ポリゴン) ---")
    print(" 1. マウスで「閉塞部」の頂点を順にクリックしてください。")
    print(" 2. [u] キーで、直前のクリックを取り消せます。")
    print(" 3. [Enter] キーで、ポリゴンを閉じて決定します。")
    
    h, w, _ = frame.shape
    scale = min(1.0, MAX_DISPLAY_HEIGHT / h)
    preview_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    
    points_preview = [] 
    window_name = "STEP 2: Draw Polygon Collision Zone (ENTER to finish)"
    cv2.namedWindow(window_name)
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points_preview.append((x, y))
            
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        draw_frame = preview_frame.copy()
        if len(points_preview) > 0:
            cv2.polylines(draw_frame, [np.array(points_preview)], isClosed=False, color=(0, 165, 255), thickness=2) # オレンジ色
        cv2.imshow(window_name, draw_frame)
        key = cv2.waitKey(20) & 0xFF
        
        if key == 13: # Enter
            if len(points_preview) >= 3:
                print("ポリゴンを決定しました。")
                break
            else:
                print("エラー: 最低3つの頂点が必要です。")
        elif key == ord('u'): # Undo
            if points_preview: points_preview.pop()
        elif key == ord('q'): # Quit
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()
    
    points_original = [ (int(p[0] / scale), int(p[1] / scale)) for p in points_preview ]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points_original)], 255)
    
    return mask # 衝突ゾーンマスクを返す

# --- (run_calibration 関数は v23.4 と同じ) ---
def run_calibration(video_path, known_length):
    points_clicked = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points_clicked) < 2:
                points_clicked.append((x, y))
                print(f"  -> 点 {len(points_clicked)} をクリック: {(x, y)}")
            else:
                print("  -> 既に2点クリック済みです。[r]でリセット, [Enter]で決定してください。")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * ROI_FRAME_TIME_S))
    ret, frame = cap.read()
    cap.release()
    if not ret: return None

    print("\n--- ピクセル/mm比 キャリブレーション ---")
    print(f"「{known_length}mmの紙」の【始点】と【終点】を順にクリックしてください。")
    print("  [r] キー: リセット, [Enter] キー: 決定")
    
    h, w, _ = frame.shape
    scale = min(1.0, MAX_DISPLAY_HEIGHT / h)
    preview_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    
    window_name = "Select 100mm Endpoints (Click 2 points, then Enter)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        draw_frame = preview_frame.copy()
        if len(points_clicked) == 1:
            cv2.circle(draw_frame, points_clicked[0], 5, (0, 0, 255), -1)
        elif len(points_clicked) == 2:
            cv2.circle(draw_frame, points_clicked[0], 5, (0, 0, 255), -1)
            cv2.circle(draw_frame, points_clicked[1], 5, (0, 0, 255), -1)
            cv2.line(draw_frame, points_clicked[0], points_clicked[1], (0, 255, 0), 2)
        
        cv2.imshow(window_name, draw_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13: # Enter
            if len(points_clicked) == 2:
                print("2点を決定しました。")
                break
            else:
                print("  -> エラー: 2点クリックしてください。")
        elif key == ord('r'): # Reset
            points_clicked = []
            print("  -> クリックをリセットしました。")
            
    cv2.destroyAllWindows()
    
    p1_orig = (int(points_clicked[0][0] / scale), int(points_clicked[0][1] / scale))
    p2_orig = (int(points_clicked[1][0] / scale), int(points_clicked[1][1] / scale))
    
    pixel_length = np.sqrt((p2_orig[0] - p1_orig[0])**2 + (p2_orig[1] - p1_orig[1])**2)
    px_per_mm = pixel_length / known_length
    
    print(f"  ピクセル長: {pixel_length:.2f} px, 既知の長さ: {known_length} mm")
    print(f"  ★ ピクセル/mm比: {px_per_mm:.4f} ★")
    return px_per_mm

# --- (run_collision_detection 関数は v23.4 と同じ) ---
def run_collision_detection(video_path, output_dir, tube_mask, collision_mask, catheter_thresh):
    """
    v23.4: ゾーン内のピクセル侵入のみを検出する
    """
    print(f"\n--- 処理中: {os.path.basename(video_path)} ---")
    base = os.path.splitext(os.path.basename(video_path))[0]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(" -> エラー: 動画ファイルが開けませんでした。")
        return None, 30.0 

    fps = cap.get(cv2.CAP_PROP_FPS)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    csv_path = os.path.join(output_dir, f"{base}_collision_status.csv")
    final_preview_path = os.path.join(output_dir, f"{base}_collision_preview.avi")

    out_final = cv2.VideoWriter(final_preview_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (vid_w, vid_h))

    frame_number = 0
    
    # マスクの輪郭をデバッグ描画用に見つけておく
    tube_contours, _ = cv2.findContours(tube_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    collision_contours, _ = cv2.findContours(collision_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f" -> 固定閾値 {catheter_thresh} (INV) を使用します。")
    collision_detected_frame = -1 

    with open(csv_path, 'w', newline='') as f:
        f.write("frame,is_collided\n")
        while True:
            ret, frame = cap.read()
            if not ret: break
            output_frame_final = frame.copy()
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, binary_full = cv2.threshold(gray_full, catheter_thresh, 255, cv2.THRESH_BINARY_INV)
            
            # 1. チューブ内部のみを抽出
            binary_in_tube = cv2.bitwise_and(binary_full, tube_mask)
            
            # 2. ★ 衝突ゾーン(ポリゴン)内のピクセルを抽出 ★
            binary_in_collision_zone = cv2.bitwise_and(binary_in_tube, collision_mask)
            
            # 3. 衝突判定
            white_pixels_in_zone = cv2.countNonZero(binary_in_collision_zone)
            
            is_collided = white_pixels_in_zone > COLLISION_PIXEL_THRESHOLD
            
            if is_collided and collision_detected_frame == -1:
                collision_detected_frame = frame_number
            
            f.write(f"{frame_number},{1 if is_collided else 0}\n")
            
            # --- 可視化 ---
            # チューブマスク (青)
            cv2.drawContours(output_frame_final, tube_contours, -1, (255, 0, 0), 2)
            
            # 衝突ゾーン (赤 / 緑)
            color = (0, 0, 255) if is_collided else (0, 255, 0) # 衝突時:赤, 通常時:緑
            cv2.drawContours(output_frame_final, collision_contours, -1, color, 2)
            
            out_final.write(output_frame_final)
            frame_number += 1
            
    cap.release()
    out_final.release()
    print(" -> フェーズ1 処理完了。")
    print(f"  衝突ステータスCSV: {csv_path}")
    print(f"  プレビュー動画: {final_preview_path}")
    
    if collision_detected_frame != -1:
        print(f"--- ★ 衝突を検出 (Frame: {collision_detected_frame}) ★ ---")
    else:
        print("--- 衝突は検出されませんでした ---")
        
    return csv_path, fps


# --- (メイン実行ブロックは v23.4 と同じ) ---
if __name__ == "__main__":
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 解析対象の全動画をリストアップ
    video_files_all = sorted(glob.glob(os.path.join(INPUT_DIR, "*.avi")))
    
    if not video_files_all:
        print(f"エラー: {INPUT_DIR} にAVI動画が見つかりません。")
        exit()

    # 2. ユーザーに動画を選択させる
    print("\n--- 解析対象の動画ファイル ---")
    for i, video_path in enumerate(video_files_all):
        print(f"  [{i+1}] {os.path.basename(video_path)}")
    print("-------------------------------")

    input_str = input("解析したい動画の番号をカンマ(,)区切りで入力してください (例: 1,3,5) (すべて選択する場合は 'all'): ").strip().lower()
    
    video_files_to_process = []
    if input_str == 'all':
        video_files_to_process = video_files_all
        print(f"-> 全 {len(video_files_to_process)} 件の動画を処理します。")
    else:
        try:
            indices = [int(s.strip()) - 1 for s in input_str.split(',')]
            for i in indices:
                if 0 <= i < len(video_files_all):
                    video_files_to_process.append(video_files_all[i])
                else:
                    print(f"警告: 番号 {i+1} は無効です。スキップします。")
        except ValueError:
            print("エラー: 無効な入力です。処理を中断します。")
            exit()

    if not video_files_to_process:
        print("動画が選択されませんでした。処理を終了します。")
        exit()

    print(f"\n--- 以下の {len(video_files_to_process)} 件の動画を処理します ---")
    for p in video_files_to_process:
        print(f"  {os.path.basename(p)}")
    print("="*50)
    
    reference_video_path = video_files_to_process[0] 

    # --- 3. 共通設定 (マスク、ROI、px/mm比) の読み込み/作成 ---
    
    # 3a. チューブマスク
    mask_save_path = os.path.join(OUTPUT_DIR, "polygon_mask.npy")
    tube_mask = None
    use_saved_mask = False
    
    if os.path.exists(mask_save_path):
        answer = input(f"記憶された「チューブマスク」を使用しますか？ (Y/n): ").strip().lower()
        if answer == '' or answer == 'y':
            print(f"--- 記憶されたチューブマスクを読み込みます ---")
            tube_mask = np.load(mask_save_path)
            use_saved_mask = True
    
    if not use_saved_mask:
        print(f"--- 新しいチューブマスクを作成します (基準: {os.path.basename(reference_video_path)}) ---")
        tube_mask, frame_size = select_polygon_roi(reference_video_path, window_title="STEP 1: Draw Polygon ROI (Tube Interior)")
        if tube_mask is not None:
            np.save(mask_save_path, tube_mask)
        else:
            print("マスクが作成されなかったため、処理を中断しました。")
            exit()

    # 3b. 衝突ゾーンROI (ポリゴン版)
    collision_mask_path = os.path.join(OUTPUT_DIR, "collision_mask.npy") # ★ .npy
    collision_mask = None
    use_saved_roi = False

    if os.path.exists(collision_mask_path):
        answer = input(f"記憶された「衝突ゾーンマスク」を使用しますか？ (Y/n): ").strip().lower()
        if answer == '' or answer == 'y':
            print(f"--- 記憶された衝突ゾーンマスクを読み込みます ---")
            collision_mask = np.load(collision_mask_path)
            use_saved_roi = True

    if not use_saved_roi:
        print(f"--- 新しい衝突ゾーンマスクを作成します (基準: {os.path.basename(reference_video_path)}) ---")
        collision_mask = select_polygon_collision_zone(reference_video_path) # ★ 呼び出す関数を変更
        if collision_mask is not None:
            np.save(collision_mask_path, collision_mask) # ★ .npy で保存
        else:
            print("衝突ゾーンが作成されなかったため、処理を中断しました。")
            exit()
            
    # 3c. ピクセル/mm比
    px_mm_path = os.path.join(OUTPUT_DIR, "px_per_mm.npy")
    px_per_mm = None
    use_saved_px_mm = False

    if os.path.exists(px_mm_path):
        px_per_mm_saved = np.load(px_mm_path)
        answer = input(f"記憶された「ピクセル/mm比 ({px_per_mm_saved[0]:.4f})」を使用しますか？ (Y/n): ").strip().lower()
        if answer == '' or answer == 'y':
            print(f"--- 記憶された px/mm比 を読み込みます ---")
            px_per_mm = px_per_mm_saved[0]
            use_saved_px_mm = True

    if not use_saved_px_mm:
        print(f"--- 新しい px/mm比 を計算します (基準: {os.path.basename(reference_video_path)}) ---")
        px_per_mm = run_calibration(reference_video_path, KNOWN_LENGTH_MM)
        if px_per_mm is not None:
            np.save(px_mm_path, [px_per_mm])
        else:
            print("px/mm比が計算されなかったため、処理を中断しました。")
            exit()

    # --- 4. ループで「選択された動画」を解析 ---
    if tube_mask is not None and collision_mask is not None and px_per_mm is not None:
        print("\n" + "="*50)
        print("--- フェーズ: 選択された動画の衝突検出を開始 ---")
        
        for video_path in video_files_to_process: 
            run_collision_detection(
                video_path, 
                OUTPUT_DIR, 
                tube_mask, 
                collision_mask, # ★ マスクを渡す
                MANUAL_THRESHOLD
            )
    
    print("\n--- 全ワークフロー完了 ---")