#coding=utf-8
import time,os,sys,shutil,pywinauto,ctypes,pygetwindow
import numpy as np
import pyautogui
import cv2
import logging
import traceback

INFO = None
VERSION = 'v1.1'
LOCATION = None
WINDOWS_NAME = None

# 解决pyinstaller打包找不到静态文件问题
def resource_path(relative_path, debug=False):
    """ Get absolute path to resource, works for dev and for PyInstaller """

    if debug:
        base_path = os.path.abspath("./temp/")
        return os.path.join(base_path, relative_path)
    else:
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)

def check_window_exist(window_title):
    for window in pygetwindow.getAllTitles():
        if window.lower() == window_title.lower():
            return True
    return False

# 确保存在文件夹
if not os.path.exists('temp'):
    os.mkdir('temp')

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

# Create a FileHandler and set its level to ERROR (or higher)
file_handler = logging.FileHandler(resource_path('error.log', debug=True))
file_handler.setLevel(logging.ERROR)

# Create a Formatter for formatting the log messages
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the FileHandler to the logger
logger.addHandler(file_handler)


class AutoFishing():

    # 读取图像，解决imread不能读取中文路径的问题
    def cv_imread(self, filePath):
        # 核心就是下面这句，一般直接用这句就行，直接把图片转为mat数据
        cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
        # imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
        # cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
        return cv_img

    def get_windows_location(self):
        try:
            app = pywinauto.Application().connect(title_re=WINDOWS_NAME)
            
            # 获取主窗口
            main_window = app.window(title=WINDOWS_NAME)

            # 将窗口置顶
            main_window.set_focus()

            main_window.topmost = True

            window = app.top_window()
            left, right, top, down  = window.rectangle().left, window.rectangle().right, window.rectangle().top, window.rectangle().bottom
            # print(f"The window position is ({left}, {right}, {top}, {down})")
            return left, right, top, down
        except pywinauto.findwindows.ElementNotFoundError:
            print(traceback.format_exc())

        except TypeError:
            print(traceback.format_exc())
        

    def get_keypoint_bounds(self,kp):
        # 获取关键点的中心坐标和直径大小
        x, y = kp.pt
        diameter = kp.size

        # 计算关键点的左上角和右下角坐标
        x1 = int(x - diameter/2)
        y1 = int(y - diameter/2)
        x2 = int(x + diameter/2)
        y2 = int(y + diameter/2)

        return x1, y1, x2, y2


    # 获取大图里面的小图坐标（简单匹配，计算量少，速度快）
    def get_simple_xy(self, img_model_path,name=None, show=False, output=True):
        """
        用来判定游戏画面的点击坐标
        :param img_model_path:用来检测的图片
        :return:以元组形式返回检测到的区域中心的坐标
        """

        # 确保存在文件夹
        if not os.path.exists('temp'):
            os.mkdir('temp')

        # 将图片截图并且保存
        pyautogui.screenshot().save("./temp/screenshot.png")
        # 待读取图像
        img = self.cv_imread("./temp/screenshot.png")

        # 裁剪图片
        img, cropped_X, cropped_Y = self.cropped_images(img)
        if cropped_X == False:
            return False, False

        # 图像模板
        img_terminal = self.cv_imread(img_model_path)
        # 读取模板的高度宽度和通道数
        height, width, channel = img_terminal.shape
        # 使用matchTemplate进行模板匹配（标准平方差匹配）
        result = cv2.matchTemplate(img, img_terminal, cv2.TM_SQDIFF_NORMED)

        # -----------------匹配最优对象-----------------

        # 解析出匹配区域的左上角图标
        upper_left = cv2.minMaxLoc(result)[2]
        # 计算出匹配区域右下角图标（左上角坐标加上模板的长宽即可得到）
        lower_right = (upper_left[0] + width, upper_left[1] + height)
        # 计算坐标的平均值并将其返回
        avg = (int((upper_left[0] + lower_right[0]) / 2) + cropped_X, int((upper_left[1] + lower_right[1]) / 2) + cropped_Y)

        # 展示图片
        if show:
            cv2.rectangle(img, upper_left, lower_right, (0, 0, 255), 2)
            cv2.imshow('img', img)
            cv2.waitKey()

        # 输出图片
        if output:
            cv2.rectangle(img, upper_left, lower_right, (0, 0, 255), 2)
            if name != None:
                # 中文路径输出
                cv2.imencode('.png', img)[1].tofile(f'./temp/{name}.png')
            else:
                cv2.imwrite(f'./temp/get_xy.png', img)
        # ---------------------------------------------
        return avg,img
    
    # 裁剪图片
    def cropped_images(self, img_rgb):

        try:
            # 获取指定窗口坐标并置顶窗口
            left, right, top, down = self.get_windows_location()

            # ---------------裁剪图片------------------

            # print(img_rgb.shape)

            # cropped = img_rgb[top+300:down-800,left+400:right-300]  # 裁剪坐标为[y0:y1, x0:x1]
            cropped = img_rgb[top:down,left:right]  # 裁剪坐标为[y0:y1, x0:x1]

            height, width = cropped.shape[:2]

            # print((height, width))

            # 计算中间区域的左上角X和Y坐标以及其宽度和高度
            cropWidth = int(width * 0.55)
            cropHeight = int(height * 0.9)
            cropped_X = (width - cropWidth) // 2
            cropped_Y = (height - cropHeight) // 2

            # print()
            # print((cropWidth, cropHeight))
            # print((cropped_X, cropped_Y))

            cropped = cropped[cropped_Y:cropped_Y+cropHeight, cropped_X:cropped_X+cropWidth]

            cv2.imwrite("./temp/screenshot_cropped.png", cropped)
            img_rgb = cropped

            # 返回裁剪图片，x偏移量，y偏移量
            return img_rgb, cropped_X + left, cropped_Y + top
        except Exception as error:
            print()
            print(repr(error))
            print()
            logger.error(traceback.format_exc())
            INFO = '没有匹配目标, 请确保游戏窗口没有脱离显示器屏幕'
            return False, False, False

    # 获取大图里面的小图坐标（精准匹配，针对简单匹配无法解决的问题，计算量多，性能慢）
    def get_precise_xy(self, img_model_path,name=None,show=False, output=True):

        """
        用来判定游戏画面的点击坐标
        :param img_model_path:用来检测的图片
        :return:以元组形式返回检测到的区域中心的坐标
        """
        # 确保存在文件夹
        if not os.path.exists('temp'):
            os.mkdir('temp')

        # 将图片截图并且保存
        pyautogui.screenshot().save("./temp/screenshot.png")

        # 待读取图像
        img_rgb = cv2.imread("./temp/screenshot.png")

        
        img_rgb, cropped_X, cropped_Y = self.cropped_images(img_rgb)
        if cropped_X == False:
            return False,False

        # 图像模板
        template = self.cv_imread(img_model_path)

        # 将图像转换为灰度图像
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # 初始化SIFT检测器
        sift = cv2.SIFT_create()

        # 在模板图像中检测特征点和描述符
        kp1, des1 = sift.detectAndCompute(template_gray, None)

        # 在原始图像中检测特征点和描述符
        kp2, des2 = sift.detectAndCompute(img_gray, None)

        # 初始化FLANN匹配器
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # 使用FLANN匹配器进行匹配
        matches = flann.knnMatch(des1, des2, k=2)

        # 过滤匹配点
        good_matches = []
        for number in [0.3]:
            for m, n in matches:
                if m.distance < number * n.distance:
                    good_matches.append(m)

        # 绘制匹配结果
        img_matches = cv2.drawMatches(template, kp1, img_rgb, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


        # 获取所有好的匹配点在原始图像中的中心坐标
        sum_x = 0
        sum_y = 0
        for match in good_matches:
            # 获取关键点在原始图像中的索引
            index = match.trainIdx

            # 获取关键点的左上角和右下角坐标
            x1, y1, x2, y2 = self.get_keypoint_bounds(kp2[index])

            # 计算中心坐标
            mid_x = int((x1 + x2) / 2)
            mid_y = int((y1 + y2) / 2)

            # 累加中心坐标
            sum_x += mid_x
            sum_y += mid_y

            # 获取关键点坐标
            x1, y1 = kp1[match.queryIdx].pt
            x2, y2 = kp2[match.trainIdx].pt

            # 绘制红色边框
            cv2.rectangle(img_rgb, (int(x2)-5, int(y2)-5), (int(x2)+5, int(y2)+5), (0, 0, 255), 2)

        # 展示图片
        if show:
            # cv2.rectangle(img, upper_left, lower_right, (0, 0, 255), 2)
            cv2.imshow('img', img_matches)
            cv2.waitKey()

        # 输出图片
        if output:

            if name != None:
                # 中文路径输出
                cv2.imencode('.png', img_matches)[1].tofile(f'./temp/{name}.png')
            else:
                cv2.imwrite(f'./temp/get_precise_xy.png', img_matches)
        # ---------------------------------------------

        # 计算平均中心坐标
        if len(good_matches) == 0:
            return False,False
        else:
            avg_x = int(sum_x / len(good_matches)) + cropped_X
            avg_y = int(sum_y / len(good_matches)) + cropped_Y

            return (avg_x, avg_y), img_matches


    # 自动点击
    def auto_click(self, var_avg):
        """
        输入一个元组，自动点击
        :param var_avg: 坐标元组
        :return: None
        """
        self.get_windows_location()
        pyautogui.click(var_avg[0], var_avg[1], button='left')
        time.sleep(1)

    # 根据指定图片进行点击
    def click_routine_images(self,img_model_path, name):
        global INFO
        
        avg,_ = self.get_precise_xy(img_model_path,name)

        if avg != False:
            INFO = f"{name}"
            # print(INFO)
            self.auto_click(avg)
            return True,avg
        else:
            avg,_ = self.get_simple_xy(img_model_path,name)
            if avg != False:
                INFO = f"{name}"
                # print(INFO)
                self.auto_click(avg)
                return True,avg
        
        return False,avg

    # 识别黑色背景的鱼（精准匹配）
    def precise_click_fish_images(self, show=False, output=True):

        global INFO

        # 确保存在文件夹
        if not os.path.exists('temp'):
            os.mkdir('temp')

        # 待读取图像
        img_matches = None

        # 最终结果
        result_list = {}

        for index in range(1,11):

            # 黑色背景鱼图片
            
            fish_black_image_path = resource_path(f'images/fish_black/{index}.png', debug=True)
            
            result, img_matches= self.get_precise_xy(fish_black_image_path)

            if result != False:
                result_list[result[0]] = index
                # print(fish_black_image_path)
        
        # 排序（确定左和右）
        for index in sorted(result_list.keys()):
       
            fish_image_path = resource_path(f'images/fish/{result_list[index]}.png',debug=True)
            avg,_ = self.get_precise_xy(fish_image_path)

            # INFO = f"正在点击图片{fish_image_path}"
            # print(INFO)

            self.auto_click(avg)
            pyautogui.moveTo(avg[0]+300, avg[1])


        # 展示图片
        if show:
            if result_list != {}:
                cv2.imshow('img', img_matches)
                cv2.waitKey()
        
        # 输出图片
        if output:
            if result_list != {}:
                cv2.imwrite('./temp/click_fish_images.png', img_matches)

        INFO = f"点击鱼完毕"
        # print(INFO)

        if result_list == {}:
            return False
        else:
            INFO = f"点击鱼完毕"
             # print(INFO)
            return True
        
        # -------------------------------------------

    # 识别黑色背景的鱼（简单匹配）
    def simple_click_fish_images(self, show=False, output=True):

        global INFO

        # 确保存在文件夹
        if not os.path.exists('temp'):
            os.mkdir('temp')

        # 将图片截图并且保存
        pyautogui.screenshot().save("./temp/screenshot.png")
        # 待读取图像
        img = self.cv_imread("./temp/screenshot.png")

        img, _, _ = self.cropped_images(img)

        # 最终结果
        result_list = {}

        for index in range(1,11):

            # 黑色背景鱼图片
            
            fish_black_image_path = resource_path(f'images/fish_black/{index}.png', debug=True)
            # 图像模板
            img_terminal = self.cv_imread(fish_black_image_path)
            # 读取模板的高度宽度和通道数
            height, width, channel = img_terminal.shape
            # 使用matchTemplate进行模板匹配（标准相关匹配）
            result = cv2.matchTemplate(img, img_terminal, cv2.TM_CCOEFF_NORMED)

            # -----------------匹配多个对象-----------------
            # 设置阈值
            threshold = 0.9
            loc = np.where(result >= threshold)

            # 遍历所有匹配结果，获取矩形区域坐标并存储在一个列表中
            rects = []
            for pt in zip(*loc[::-1]):
                rects.append((pt[0], pt[1], img_terminal.shape[1], img_terminal.shape[0]))

            # 绘制矩形框和匹配结果
            for rect in rects:
                cv2.rectangle(img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 0, 255), 2)
            
            if rects != []:
                result_list[rects[0][0]] = index
                # print(fish_black_image_path)
        
        # 排序（确定左和右）
        for index in sorted(result_list.keys()):

            fish_image_path = resource_path(f'images/fish/{result_list[index]}.png',debug=True)
            avg,_ = self.get_simple_xy(fish_image_path)
 
            # INFO = f"正在点击图片{fish_image_path}"
            # print(INFO)

            self.auto_click(avg)
            pyautogui.moveTo(avg[0]+300, avg[1])


        # 展示图片
        if show:
            cv2.imshow('img', img)
            cv2.waitKey()
        
        # 输出图片
        if output:
            cv2.imwrite('./temp/click_fish_images.png', img)

        if result_list == {}:
            return False
        else:
            INFO = f"点击鱼完毕"
             # print(INFO)
            return True
       
        # -------------------------------------------

    def auto_fish(self):

        global LOCATION
         
        # 开始钓鱼-1
        status,avg = self.click_routine_images("./images/core/1.png", "开始钓鱼")
        if status:
            LOCATION = avg
            pyautogui.moveTo(avg[0]+200, avg[1])

        # 取消钓鱼-2
        status,avg = self.click_routine_images("./images/core/2.png", "取消钓鱼")
        if status:
            LOCATION = avg
            pyautogui.moveTo(avg[0]+200, avg[1])

        # 渔网确认-3
        self.click_routine_images("./images/core/3.png", "渔网确认")

        # 传送我的房间-4
        status,_ = self.click_routine_images("./images/core/4.png", "传送我的房间")
        if status:
            # 点击鱼-5
            status = self.simple_click_fish_images()

            # 如果简单匹配失效，采用精准匹配
            if not status:
                self.precise_click_fish_images()

            time.sleep(1)
            pyautogui.press('esc')

            INFO = f"回收渔网完毕"
            # print(INFO)

        # 开始钓鱼-1
        self.click_routine_images("./images/core/1.png", "开始钓鱼")


#coding=utf-8
import datetime,re,threading,PIL,webbrowser,json,subprocess
from time import sleep
from tkinter import Label,Frame,messagebox,Entry,Tk,Text,Button,Toplevel,StringVar,END
from tkinter import messagebox
from tkinter.ttk import Combobox
from PIL import ImageTk

class WinGUI(Tk):
    def __init__(self):
        super().__init__()
        self.__win()
        self.loop_time = self.__init_loop_time() * 60
        self.thread = None
        self.tk_text_lb4no8xs = self.__tk_text_lb4no8xs()
        self.tk_button_start = self.__tk_button_start()
        self.tk_button_stop = self.__tk_button_stop()
        self.tk_button_tutorial = self.__tk_button_tutorial()
        self.tk_label_minute = self.__tk_label_minute()
        self.tk_entry_minute = self.__tk_entry_minute()

    def __init_loop_time(self):

        # 确保存在文件夹
        if not os.path.exists('temp'):
            os.mkdir('temp')

        # 复制exe的images文件夹出来
        if os.path.exists('./temp/images'):
            # 如果目标文件夹已经存在，删除它
            # shutil.rmtree('./temp/images')
            pass
        else: 
            shutil.copytree(resource_path('images'), './temp/images')


        if not os.path.exists('./temp/settings.json'):
            with open('./temp/settings.json','w',encoding='utf-8') as f:
                f.write(json.dumps(dict(loop_time=60), ensure_ascii=False))
            return 60
        else:
            with open('./temp/settings.json','r',encoding='utf-8') as f:
                loop_time = json.loads(f.read()).get('loop_time')
                # print(loop_time)
                if loop_time != None:
                    return loop_time
                else:
                    return 60
    
    def __win(self):
        global VERSION
        self.title(f"自动钓鱼收网 {VERSION}")
        # 设置窗口大小、居中
        width = 420
        height = 200
        screenwidth = self.winfo_screenwidth()
        screenheight = self.winfo_screenheight()
        geometry = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        self.geometry(geometry)
        self.resizable(width=False, height=False)

    def __tk_entry_minute(self):

        entry = Entry(self)
        entry.insert(0, self.__init_loop_time())
        entry.place(x=130, y=10, width=75)
        return entry

    def __tk_label_minute(self):
        label = Label(self, text="几分钟收网一次：")
        label.place(x=10, y=10)
        return label
    
    def __tk_text_lb4no8xs(self):
        text = Text(self)
        text.tag_config('green', font=('黑体', 10, 'bold') ,foreground='green')
        text.tag_config('red', font=('黑体', 10, 'bold'), foreground='red')
        text.place(x=10, y=45, width=400, height=143)
        return text

    def __tk_button_start(self):
        btn = Button(self, text="开始")
        btn.place(x=217, y=6, width=60, height=30)
        return btn

    def __tk_button_stop(self):
        btn = Button(self, text="停止")
        btn.place(x=283, y=6, width=60, height=30)
        return btn

    def __tk_button_tutorial(self):
        btn = Button(self, text="教程")
        btn.place(x=350, y=6, width=60, height=30)
        return btn

class Win(WinGUI):
    flag = False

    def __init__(self):
        super().__init__()
        self.__event_bind()
        self.tutorial_image = ImageTk.PhotoImage(PIL.Image.open(resource_path('images/core/status.png')))
        self.tutorial_two_image_1 = ImageTk.PhotoImage(PIL.Image.open(resource_path('images/core/1.png')))
        self.tutorial_two_image_2 = ImageTk.PhotoImage(PIL.Image.open(resource_path('images/core/2.png')))

    # 读取图像，解决imread不能读取中文路径的问题
    def cv_imread(self, filePath):
        # 核心就是下面这句，一般直接用这句就行，直接把图片转为mat数据
        cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
        # imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
        # cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
        return cv_img
    
    def __tk_entry_minute_constraint(self):

        # 确保存在文件夹
        if not os.path.exists('temp'):
            os.mkdir('temp')

        pattern = r"^[1-9]*[1-9][0-9]*$"
        match = re.match(pattern, self.tk_entry_minute.get())

        if match:
            loop_time = int(match.group())
            self.loop_time = loop_time * 60
            with open('./temp/settings.json','w',encoding='utf-8') as f:
                f.write(json.dumps(dict(loop_time=loop_time), ensure_ascii=False))

            return True
        else:
            messagebox.showwarning('警告','请输入大于 0 的数字')
            return False

    def __open_url(self, url):
        webbrowser.open_new(url)

    # 打开文件夹目录
    def __open_folder(self):
        path = os.getcwd()
        if path:
            # Windows系统下使用explorer打开目录，MacOS和Linux系统下使用open命令打开目录
            if os.name == 'nt':
                subprocess.run(['explorer', resource_path('images',debug=True)])
            elif os.name == 'posix':
                subprocess.run(['open', resource_path('images',debug=True)])

    def __button_tutorial_window(self):
        # 创建一个新窗口
        new_window = Toplevel()

        # 设置窗口大小、居中
        width = 830
        height = 1260
        screenwidth = self.winfo_screenwidth()
        screenheight = self.winfo_screenheight()
        geometry = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        new_window.geometry(geometry)
        new_window.resizable(width=False, height=False)

        ico_path = resource_path('icons/talesrunner.ico')
        new_window.iconbitmap(ico_path)

        # 程序前置条件说明
        tutorial_title_label = Label(new_window, text='启动程序前置条件', font=('黑体', 25, 'bold'), foreground='red', anchor='center', justify='center')
        tutorial_title_label.pack(pady=10)

        tutorial_title_label = Label(new_window, text='①管理员权限运行\n②满足以下任意一种钓鱼状态', font=('黑体', 18, 'bold'), anchor='center', justify='center')
        tutorial_title_label.pack(pady=10)

        # 钓鱼状态说明
        tutorial_images_label = Label(new_window, image=self.tutorial_image, anchor='center', justify='center')
        tutorial_images_label.pack(pady=10)

        # 程序前置条件说明补充
        tutorial_title_two_label = Label(new_window, text='③鼠标调整好钓鱼角度\n本程序基于“彩色”图像识别\n周围少一点“绿色”“紫色”建筑物\n否则影响“钓鱼图标”识别的准确率', font=('黑体', 18, 'bold'), anchor='center', justify='center')
        tutorial_title_two_label.pack(pady=10)

        # 钓鱼状态说明补充
        tutorial_images_two_frame = Frame(new_window)
        tutorial_images_two_frame.pack(pady=10)
        tutorial_images_two_frame_left = Frame(tutorial_images_two_frame)
        tutorial_images_two_frame_left.pack(side='left')
        tutorial_images_two_frame_right = Frame(tutorial_images_two_frame)
        tutorial_images_two_frame_right.pack(side='right')

        tutorial_images_two_label = Label(tutorial_images_two_frame_left, image=self.tutorial_two_image_1, anchor='center', justify='center')
        tutorial_images_two_label.pack()
        tutorial_images_two_label = Label(tutorial_images_two_frame_right, image=self.tutorial_two_image_2, anchor='center', justify='center')
        tutorial_images_two_label.pack()

        tutorial_title_two_label = Label(new_window, text='④点击“开始”即可钓鱼收网', font=('黑体', 18, 'bold'), foreground='red', anchor='center', justify='center')
        tutorial_title_two_label.pack(pady=10)

        tutorial_title_two_label = Label(new_window, text='如果出现无法识别的情况\n1.手动调整“游戏分辨率”(1280x960,1280x1024等等)\n2.手动截图“新图标”替换“默认的图标”', font=('黑体', 18, 'bold'), anchor='center', justify='center')
        tutorial_title_two_label.pack(pady=10)

        images_button = Button(new_window, text='点击打开钓鱼图标目录, 修改成自己的图标', command=self.__open_folder, font=('黑体', 20, 'bold'), foreground='red', background="white")
        images_button.pack(pady=30)

        # 项目信息
        frame = Frame(new_window)
        frame.pack(pady=10)
        frame_left = Frame(frame)
        frame_left.pack(side='left')
        frame_right = Frame(frame)
        frame_right.pack(side='right')

        github_label = Label(frame_left, text='GitHub仓库', font=('黑体', 15, 'bold','underline'), foreground='blue', anchor='w', underline=True)
        github_label.pack(padx=20)
        github_label.bind("<Button-1>", lambda event: self.__open_url("https://github.com/mochazi/AutoFishing"))

        github_release_label = Label(frame_right, text='最新版本', font=('黑体', 15, 'bold','underline'), foreground='blue', anchor='w')
        github_release_label.pack(padx=20)
        github_release_label.bind("<Button-1>", lambda event: self.__open_url("https://github.com/mochazi/AutoFishing/releases"))

    def __event_bind(self):
        self.tk_button_start.config(command=self.start)
        self.tk_button_stop.config(command=self.stop)
        self.tk_button_tutorial.config(command=self.__button_tutorial_window)

    def start(self):
        if WINDOWS_NAME:
            self.flag = True
            if self.__tk_entry_minute_constraint():
                self.thread = threading.Thread(target=self.print_info)
                self.thread.setDaemon(True) 
                self.thread.start()
        else:
            messagebox.showwarning('警告','没有找到游戏窗口\n\n请使用窗口化而不是无边框')

    def stop(self):
        self.flag = False
        INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + '钓鱼任务已停止'
        self.tk_text_lb4no8xs.insert(END, "\n" + INFO + "\r\n", 'red')
        INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + '如果上一轮任务没结束, 将完成后再停止'
        self.tk_text_lb4no8xs.insert(END, "\n" + INFO + "\r\n\n", 'red')
        self.tk_text_lb4no8xs.see(END)
        

    def print_info(self):

        global INFO,LOCATION

        while self.flag:
            
            try:
                auto_fishing = AutoFishing()

                # 置顶窗口，关闭多余窗口，开始钓鱼
                auto_fishing.get_windows_location()
                pyautogui.press('esc')
                time.sleep(1)
                pyautogui.press('esc')

                # 开始钓鱼-1
                status,avg = auto_fishing.click_routine_images(resource_path('images/core/1.png', debug=True), "开始钓鱼")
                if status:
                    INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + INFO
                    self.tk_text_lb4no8xs.insert(END, INFO + "\r\n", 'green')
                    self.tk_text_lb4no8xs.see(END)
                    LOCATION = avg
                    pyautogui.moveTo(avg[0]+200, avg[1])

                # 取消钓鱼-2
                status,avg = auto_fishing.click_routine_images(resource_path(f'images/core/2.png', debug=True), "取消钓鱼")
                if status:
                    INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + INFO
                    self.tk_text_lb4no8xs.insert(END, INFO + "\r\n", 'green')
                    self.tk_text_lb4no8xs.see(END)
                    LOCATION = avg
                    pyautogui.moveTo(avg[0]+200, avg[1])

                # 渔网确认-3
                status,avg = auto_fishing.click_routine_images(resource_path(f'images/core/3.png', debug=True), "渔网确认")
                if status:
                    INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + INFO
                    self.tk_text_lb4no8xs.insert(END, INFO + "\r\n", 'green')
                    self.tk_text_lb4no8xs.see(END)

                # 传送我的房间-4
                status,_ = auto_fishing.click_routine_images(resource_path(f'images/core/4.png', debug=True), "传送我的房间")
                if status:
                    INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + INFO
                    self.tk_text_lb4no8xs.insert(END, INFO + "\r\n", 'green')
                    self.tk_text_lb4no8xs.see(END)

                    # 点击鱼-5
                    INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + '正在点击鱼, 需要几秒钟计算位置'
                    self.tk_text_lb4no8xs.insert(END, INFO + "\r\n", 'green')
                    self.tk_text_lb4no8xs.see(END)
                    
                    # 简单匹配
                    status = auto_fishing.simple_click_fish_images()
                    
                    # 如果简单匹配失效，采用精准匹配
                    if not status:
                        auto_fishing.precise_click_fish_images()

                    INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + INFO
                    self.tk_text_lb4no8xs.insert(END, INFO + "\r\n", 'green')
                    self.tk_text_lb4no8xs.see(END)

                    time.sleep(1)
                    pyautogui.press('esc')

                    time.sleep(1)
                    pyautogui.press('esc')

                    INFO = f"回收渔网完毕"
                    INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + INFO
                    self.tk_text_lb4no8xs.insert(END, INFO + "\r\n", 'red')
                    self.tk_text_lb4no8xs.see(END)

                # 开始钓鱼-1
                status,avg = auto_fishing.click_routine_images(resource_path('images/core/1.png', debug=True), "开始下一轮钓鱼")
                if status:
                    LOCATION = avg
                    pyautogui.moveTo(avg[0]+200, avg[1])
                    INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + INFO
                    self.tk_text_lb4no8xs.insert(END, INFO + "\r\n", 'green')
                    self.tk_text_lb4no8xs.see(END)
                else:
                    INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + '没有匹配目标, 请确保游戏窗口没有脱离显示器屏幕'
                    self.tk_text_lb4no8xs.insert(END, INFO + "\r\n", 'red')
                    self.tk_text_lb4no8xs.see(END)

                # INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + INFO
                # self.tk_text_lb4no8xs.insert(1.0, INFO + "\r\n")

                sleep(self.loop_time)
            
            except FileNotFoundError:
                logger.error(traceback.format_exc())
                INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + '钓鱼图像文件不存在, 正在重新生成文件'
                self.tk_text_lb4no8xs.insert(END, INFO + "\r\n", 'red')
                self.tk_text_lb4no8xs.see(END)

                # 复制exe的images文件夹出来
                if os.path.exists('./temp/images'):
                    # 如果目标文件夹已经存在，删除它
                    shutil.rmtree('./temp/images')
                    shutil.copytree(resource_path('images'), './temp/images')

                auto_fishing.get_windows_location()
                time.sleep(1)
                pyautogui.press('esc')
                time.sleep(1)
                pyautogui.press('esc')

                INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + '钓鱼图像文件生成完毕, 即将开始钓鱼'
                self.tk_text_lb4no8xs.insert(END, INFO + "\r\n", 'green')
                self.tk_text_lb4no8xs.see(END)
                
            except Exception as error:
                print(repr(error))
                INFO = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + '没有匹配目标, 请确保游戏窗口没有脱离显示器屏幕'
                logger.error(traceback.format_exc())
                self.tk_text_lb4no8xs.insert(END, INFO + "\r\n", 'red')
                self.tk_text_lb4no8xs.see(END)
                self.stop()

# 提升为管理员权限
def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False
    
if __name__ == "__main__":

    # 检查游戏是否存在
    for name in ["Tales Runner", "Tales Runner ver."]:
        if check_window_exist(name):
            WINDOWS_NAME = name

    if is_admin():
        win = Win()
        ico_path = resource_path('icons/talesrunner.ico')
        win.iconbitmap(ico_path)
        win.mainloop()
    else:
        # Re-run the program with admin rights
        ctypes.windll.shell32.ShellExecuteW(None,"runas", sys.executable, '', None, 1)
        
    