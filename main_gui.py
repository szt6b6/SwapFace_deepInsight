import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
import cv2
import insightface
from insightface.app import FaceAnalysis

root = tk.Tk()
root.title("换脸GUI")
root.geometry("800x600")


source_img = None
target_img = None
target_video = None
target_photo = None
face_analyer = FaceAnalysis(name='buffalo_l')
face_analyer.prepare(ctx_id=0, det_size=(640, 640))
face_swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)


def set_source_img():
    # 打开文件选择对话框，选择一张图片
    filename = filedialog.askopenfilename(initialdir=".", title="Select a image", filetypes=(("PNG files", "*.png"), 
                                                                                            ("JPEG files", "*.jpg"),
                                                                                            ("all files", "*.*")))
    if filename:
        global source_img 
        source_img = cv2.imread(filename, cv2.IMREAD_COLOR)
        label.config(text="Set source_img done. Path: %s" % filename)
    else:
        label.config(text="Set source_img failed")

def set_target_img():
    # 打开文件选择对话框，选择一张图片
    filename = filedialog.askopenfilename(initialdir=".", title="Select a image or videos", filetypes=(("PNG files", "*.png"), 
                                                                                            ("JPEG files", "*.jpg"),
                                                                                            ("all files", "*.*"),                                                                                                                                          ("VIDEOS files", ".mp4")))
    if filename:
        global target_video, target_img  
        if filename[-3:] == "mp4": # 视频文件
            target_img = None
            target_video = cv2.VideoCapture(filename)
            label.config(text="Set target_video done. Path: %s" % filename)
            return 
        target_img = cv2.imread(filename, cv2.IMREAD_COLOR)
        target_video = None
        label.config(text="Set target_img done. Path: %s" % filename)
    else:
        label.config(text="Set target_img or video failed")


def process_target_img(source_face):
    global target_img, target_photo
    target_faces = face_analyer.get(target_img)
    target_faces = sorted(target_faces, key=lambda x: x.bbox[0])

    for target_face in target_faces:
        target_img = face_swapper.get(target_img, target_face, source_face, paste_back=True)
    
    # 将OpenCV图像转换为Pillow图像
    target_ = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

    # 目标图像宽度
    height = 450

    # 计算缩放比例，保持纵横比不变
    ratio = height / target_.shape[0]
    width = int(target_.shape[1] * ratio)

    target_ = cv2.resize(target_, (width, height))
    pil_img = Image.fromarray(target_, mode='RGB')

    # 将Pillow图像转换为PhotoImage对象
    target_photo = ImageTk.PhotoImage(pil_img)
    label.config(image=target_photo)
    label.update()



def run():
    global target_img, source_img, target_video, target_photo
    if source_img is None:
        label.config(text="No source_img")
        return

    if target_img is None and target_video is None:
        label.config(text="No target_img or video")
        return
    
    if(target_img is not None):
        source_faces = face_analyer.get(source_img)
        source_faces = sorted(source_faces, key = lambda x : x.bbox[0])
        assert len(source_faces)==1
        source_face = source_faces[0]

        process_target_img(source_face=source_face)
        return
    
    if target_video is not None:
        source_faces = face_analyer.get(source_img)
        source_faces = sorted(source_faces, key = lambda x : x.bbox[0])
        assert len(source_faces)==1
        source_face = source_faces[0]

        # 获取视频帧率，尺寸等信息
        fps = target_video.get(cv2.CAP_PROP_FPS)
        width = int(target_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(target_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 设置保存视频的编码器和参数
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

        while(target_video.isOpened()):
            # 从摄像头读取一帧数据
            ret, target_img = target_video.read()
            if(ret):
                process_target_img(source_face=source_face)
                out.write(target_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        out.release()
        
        return

# 创建Frame组件
frame_button = tk.Frame(root)
frame_label = tk.Frame(root)
# 使用grid()方法设置frame1组件在第一行第一列，占用一行两列
frame_button.grid(row=0, column=1, columnspan=3, sticky="nsew")
frame_label.grid(row=1, column=1, sticky="nsew")
# 设置按钮样式
set_s_button = tk.Button(frame_button, text="选择原图片", font=("Arial", 10), fg="black",
                   command=set_source_img, padx=10, pady=5, relief="groove")
set_s_button.grid(row=1, column=1)

set_t_button = tk.Button(frame_button, text="选择目标图片或mp4视频", font=("Arial", 10), fg="black",
                   command=set_target_img, padx=10, pady=5, relief="groove")
set_t_button.grid(row=1, column=2)

run_button = tk.Button(frame_button, text="执行换脸", font=("Arial", 10), fg="black",
                   command=run, padx=10, pady=5, relief="groove")
run_button.grid(row=1, column=3)

label = tk.Label(frame_label)
label.pack(fill="both", expand=True)



if(target_video is not None):
    target_video.release()

root.mainloop()
