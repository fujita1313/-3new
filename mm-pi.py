import cv2
import numpy as np
import os

# --- ★★★ ユーザー設定 ★★★ ---
# キャリブレーションに使う動画
CALIBRATION_VIDEO_PATH = r"C:\Users\haruk\OneDrive\ドキュメント\修士\実験データ\20251022\解析結果\時間同期\カテーテル\trimmed_videos\IMG_0624_trimmed.avi"
# 基準となる紙の「既知の長さ」(ミリメートル)
KNOWN_LENGTH_MM = 100.0
# --- 設定ここまで ---


# クリックした座標を保存するリスト
points_clicked = []

def mouse_callback(event, x, y, flags, param):
    """
    マウスのクリックイベントを処理するコールバック関数
    """
    global points_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points_clicked) < 2:
            points_clicked.append((x, y))
            print(f"  -> 点 {len(points_clicked)} をクリック: {(x, y)}")
        else:
            print("  -> 既に2点クリック済みです。[r]でリセット, [Enter]で決定してください。")

def get_pixel_length(video_path, known_length):
    global points_clicked
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"エラー: 動画ファイルが開けません {video_path}")
        return None
    
    # 2秒時点のフレームを読み込む
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * 2.0))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("エラー: フレームを読み込めません")
        return None

    print("\n--- ピクセル/mm キャリブレーション (2点クリック版) ---")
    print(f"「{known_length}mmの紙」の【始点】と【終点】を順にクリックしてください。")
    print("  [r] キー: クリックをリセット")
    print("  [Enter] キー: 2点を決定")
    print("  [q] キー: 中止")
    
    h, w, _ = frame.shape
    scale = min(1.0, 800 / h)
    preview_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    
    window_name = "Select 100mm Endpoints (Click 2 points, then Enter)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        # 描画用のフレームをコピー
        draw_frame = preview_frame.copy()
        
        if len(points_clicked) == 1:
            # 1点目
            cv2.circle(draw_frame, points_clicked[0], 5, (0, 0, 255), -1) # 赤
        elif len(points_clicked) == 2:
            # 2点目
            cv2.circle(draw_frame, points_clicked[0], 5, (0, 0, 255), -1)
            cv2.circle(draw_frame, points_clicked[1], 5, (0, 0, 255), -1)
            # 2点間に線を引く
            cv2.line(draw_frame, points_clicked[0], points_clicked[1], (0, 255, 0), 2) # 緑
        
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
        elif key == ord('q'): # Quit
            cv2.destroyAllWindows()
            print("キャリブレーションを中断しました。")
            return None
            
    cv2.destroyAllWindows()
    
    # 座標を元のスケールに戻す
    p1_orig = (int(points_clicked[0][0] / scale), int(points_clicked[0][1] / scale))
    p2_orig = (int(points_clicked[1][0] / scale), int(points_clicked[1][1] / scale))
    
    # ユークリッド距離を計算
    pixel_length = np.sqrt((p2_orig[0] - p1_orig[0])**2 + (p2_orig[1] - p1_orig[1])**2)
    
    px_per_mm = pixel_length / known_length
    
    print("\n--- キャリブレーション完了 ---")
    print(f"  点1: {p1_orig}, 点2: {p2_orig}")
    print(f"  ピクセル長 (斜め): {pixel_length:.2f} px")
    print(f"  既知の長さ: {known_length} mm")
    print(f"  ピクセル/mm比: {px_per_mm:.4f}")
    print(f"\n★ この「{px_per_mm:.4f}」の値を、ステップ2のスクリプトの PX_PER_MM にコピーしてください。")
    return px_per_mm

if __name__ == "__main__":
    get_pixel_length(CALIBRATION_VIDEO_PATH, KNOWN_LENGTH_MM)