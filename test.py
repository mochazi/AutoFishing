#coding=utf-8
import time,os,cv2,pyautogui,pywinauto
import numpy as np

INFO = ''
LOCATION = None

class AutoFishing():

    # 读取图像，解决imread不能读取中文路径的问题
    def cv_imread(self, filePath):
        # 核心就是下面这句，一般直接用这句就行，直接把图片转为mat数据
        cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
        # imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
        # cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
        return cv_img

    def get_windows_location(self):
        app = pywinauto.Application().connect(title_re="Tales Runner")
        
        # 获取主窗口
        main_window = app.window(title="Tales Runner")

        # 将窗口置顶
        main_window.set_focus()
        main_window.topmost = True

        window = app.top_window()
        left, right, top, down  = window.rectangle().left, window.rectangle().right, window.rectangle().top, window.rectangle().bottom
        # print(f"The window position is ({left}, {right}, {top}, {down})")
        return left, right, top, down

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
            INFO = f"正在点击{name}"
            print(INFO)
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
            
            fish_black_image_path = f'images/fish_black/{index}.png'
            
            result, img_matches= self.get_precise_xy(fish_black_image_path)

            if result != False:
                result_list[result[0]] = index
                # print(fish_black_image_path)
        
        # 排序（确定左和右）
        for index in sorted(result_list.keys()):

            fish_image_path = f'images/fish/{result_list[index]}.png'
            avg,_ = self.get_precise_xy(fish_image_path)

            # INFO = f"正在点击图片{fish_image_path}"
            # print(INFO)

            self.auto_click(avg)


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
            
            fish_black_image_path = f'images/fish_black/{index}.png'
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

            fish_image_path = f'images/fish/{result_list[index]}.png'
            avg,_ = self.get_simple_xy(fish_image_path)

            # INFO = f"正在点击图片{fish_image_path}"
            # print(INFO)

            self.auto_click(avg)


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
         
        # 置顶窗口，关闭多余窗口，开始钓鱼
        self.get_windows_location()
        pyautogui.press('esc')
        time.sleep(1)
        pyautogui.press('esc')

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

            time.sleep(1)
            pyautogui.press('esc')

            INFO = f"回收渔网完毕"
            print(INFO)

        # 开始钓鱼-1
        status,avg = self.click_routine_images("./images/core/1.png", "开始钓鱼")
        if status:
            LOCATION = avg
            pyautogui.moveTo(avg[0]+200, avg[1])

if __name__ == '__main__':
    (
        AutoFishing().auto_fish()
    )