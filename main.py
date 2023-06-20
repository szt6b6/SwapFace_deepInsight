import cv2
import insightface
from insightface.app import FaceAnalysis

if __name__ == '__main__':
    # cudnn_ops_infer64_8.dll 把这个东西所在的目录加入到环境变量中
    face_analyer = FaceAnalysis(name='buffalo_l')
    face_analyer.prepare(ctx_id=0, det_size=(640, 640))
    face_swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)


    source_img = cv2.imread("彭于晏.png")
    source_faces = face_analyer.get(source_img)
    source_faces = sorted(source_faces, key = lambda x : x.bbox[0])
    assert len(source_faces)==1
    source_face = source_faces[0]


    cap = cv2.VideoCapture(0)  # 修改参数0为其他数字，可以切换摄像头
    while(True):
        # 从摄像头读取一帧数据
        ret, target_img = cap.read()

        # 处理目标图片
        target_faces = face_analyer.get(target_img)
        target_faces = sorted(target_faces, key=lambda x: x.bbox[0])
        for target_face in target_faces:
            target_img = face_swapper.get(target_img, target_face, source_face, paste_back=True)


        # 显示处理后的视频帧
        cv2.imshow('',target_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放视频流对象
    cap.release()
    # 关闭所有窗口
    cv2.destroyAllWindows()
