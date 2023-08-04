import cv2
import pyrealsense2 as rs
import numpy as np

def main():
    # 카메라 객체 생성
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    # 카메라 시작
    pipeline.start(config)

    try:
        while True:
            # 카메라로부터 프레임 가져오기
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            # 프레임 데이터를 NumPy 배열로 변환
            color_image = np.asanyarray(color_frame.get_data())

            # 화면에 영상 보여주기
            cv2.imshow('Realsense Camera', color_image)

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # 카메라 정리 및 창 닫기
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()