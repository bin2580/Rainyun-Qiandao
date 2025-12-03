import logging
import os
import random
import re
import time
import subprocess
import sys

import cv2
import ddddocr
import requests
from selenium import webdriver
from selenium.common import TimeoutException
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


# 修改init_selenium函数，修复linux变量作用域问题
# 在文件开头导入webdriver-manager
# 修改这两行导入语句
# from webdriver_manager.chrome import ChromeDriverManager
# from webdriver_manager.core.utils import ChromeType

# 修改为正确的导入方式
try:
    from webdriver_manager.chrome import ChromeDriverManager
    # 尝试不同的ChromeType导入路径
    try:
        from webdriver_manager.core.utils import ChromeType
    except ImportError:
        try:
            from webdriver_manager.chrome import ChromeType
        except ImportError:
            # 如果找不到ChromeType，设置为None
            ChromeType = None
except ImportError:
    print("webdriver_manager未安装，将使用备用方式")
    ChromeDriverManager = None
    ChromeType = None

# --- START OF MODIFICATION: 添加通知函数导入 ---
try:
    from notify import send
    print("已加载通知模块 (notify.py)")
except ImportError:
    print("警告: 未找到 notify.py，将无法发送通知。")
    def send(*args, **kwargs):
        pass
# --- END OF MODIFICATION ---


def init_selenium(debug=False, headless=False):
    ops = webdriver.ChromeOptions()
    
    # 无论什么环境都添加无头模式选项
    if headless or os.environ.get("GITHUB_ACTIONS", "false") == "true":
        for option in ['--headless', '--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu']:
            ops.add_argument(option)
    
    # 添加通用选项
    ops.add_argument('--window-size=1920,1080')
    ops.add_argument('--disable-blink-features=AutomationControlled')
    ops.add_argument('--no-proxy-server')
    ops.add_argument('--lang=zh-CN')
    
    # 环境变量判断是否在GitHub Actions中运行
    is_github_actions = os.environ.get("GITHUB_ACTIONS", "false") == "true"
    
    if debug and not is_github_actions:
        ops.add_experimental_option("detach", True)
    
    # 尝试不同的ChromeDriver使用策略
    # 策略1: 直接使用系统路径中的ChromeDriver（最简单可靠）
    try:
        print("尝试直接使用系统ChromeDriver...")
        # 不指定service，让Selenium自动查找系统路径中的ChromeDriver
        driver = webdriver.Chrome(options=ops)
        print("成功使用系统ChromeDriver")
        return driver
    except Exception as e:
        print(f"系统ChromeDriver失败: {e}")
    
    # 策略2: 优化webdriver-manager的使用方式
    try:
        print("尝试使用webdriver-manager...")
        if ChromeDriverManager:
            # 仅当ChromeType可用时才指定chrome_type参数
            if ChromeType and hasattr(ChromeType, 'GOOGLE'):
                manager = ChromeDriverManager(chrome_type=ChromeType.GOOGLE)
            else:
                # 在新版本中，可能不再需要指定ChromeType
                manager = ChromeDriverManager()
            
            # 获取驱动路径但不自动安装
            driver_path = manager.install()
            print(f"获取到ChromeDriver路径: {driver_path}")
            # 手动创建service并指定正确的驱动路径
            service = Service(driver_path)
            driver = webdriver.Chrome(service=service, options=ops)
            print("成功使用webdriver-manager")
            return driver
        else:
            raise ImportError("webdriver_manager未安装")
    except Exception as e:
        print(f"webdriver-manager失败: {e}")
    
    # 策略3: 作为最后的备用，尝试使用固定路径
    try:
        print("尝试使用备用ChromeDriver路径...")
        # 尝试常见的ChromeDriver路径
        common_paths = ['/usr/local/bin/chromedriver', '/usr/bin/chromedriver', './chromedriver', 'chromedriver']
        for path in common_paths:
            try:
                service = Service(path)
                driver = webdriver.Chrome(service=service, options=ops)
                print(f"成功使用备用路径: {path}")
                return driver
            except:
                continue
    except Exception as e:
        print(f"备用路径失败: {e}")
    
    # 所有策略都失败时的错误处理
    print("错误: 无法初始化ChromeDriver，请检查Chrome和ChromeDriver的安装")
    # 在GitHub Actions环境中，尝试安装ChromeDriver的备用方法
    if is_github_actions:
        print("在GitHub Actions环境中，尝试安装ChromeDriver...")
        try:
            # 使用npm的chromedriver包作为备用
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'chromedriver-binary-auto'])
            import chromedriver_binary  # 这个包会自动设置路径
            driver = webdriver.Chrome(options=ops)
            print("成功使用chromedriver-binary-auto")
            return driver
        except Exception as e:
            print(f"备用安装失败: {e}")
    
    # 彻底失败
    raise Exception("无法初始化Selenium WebDriver")

def download_image(url, filename):
    os.makedirs("temp", exist_ok=True)
    try:
        # 禁用代理以避免连接问题
        response = requests.get(url, timeout=10, proxies={"http": None, "https": None}, verify=False)
        if response.status_code == 200:
            path = os.path.join("temp", filename)
            with open(path, "wb") as f:
                f.write(response.content)
            return True
        else:
            logger.error(f"下载图片失败！状态码: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"下载图片异常: {str(e)}")
        return False


def get_url_from_style(style):
    return re.search(r'url\(["\']?(.*?)["\']?\)', style).group(1)


def get_width_from_style(style):
    return re.search(r'width:\s*([\d.]+)px', style).group(1)


def get_height_from_style(style):
    return re.search(r'height:\s*([\d.]+)px', style).group(1)


# --- START OF MODIFICATION: 增强 process_captcha 逻辑 ---
def process_captcha():
    # 最大重试次数，防止无限循环
    max_captcha_retries = 5
    current_retries = 0
    
    # 循环尝试加载验证码图片
    while current_retries < max_captcha_retries:
        try:
            # 1. 尝试下载验证码图片
            download_captcha_img() # 这个函数内部会等待 slideBg 元素出现
            
            # 2. 图片下载成功，跳出循环进行识别
            break
            
        except TimeoutException:
            current_retries += 1
            logger.error(f"获取验证码图片失败（可能是页面未加载完成），尝试第 {current_retries}/{max_captcha_retries} 次重试...")
            
            # 尝试刷新页面
            try:
                # 确保在 iframe 外操作刷新按钮
                driver.switch_to.default_content()
                # 尝试点击刷新按钮
                reload = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="reload"]')))
                time.sleep(1)
                reload.click()
                logger.info("点击刷新按钮，重新加载验证码...")
                time.sleep(5)
                # 重新切换回验证码 iframe
                wait.until(EC.frame_to_be_available_and_switch_to_it((By.ID, "tcaptcha_iframe_dy")))
            except Exception as e:
                logger.warning(f"刷新验证码时发生错误: {e}")
                
            if current_retries >= max_captcha_retries:
                # 达到最大重试次数，抛出异常
                raise Exception("多次尝试后仍无法获取验证码图片，放弃处理。")
            
            time.sleep(random.uniform(2, 4)) # 随机等待后再次尝试
        
        except Exception as e:
            logger.error(f"下载验证码图片时发生未知错误: {e}")
            raise # 遇到其他错误直接向上抛出

    # --- 原来的识别和点击逻辑 (从这里开始) ---
    try:
        if check_captcha():
            logger.info("开始识别验证码")
            captcha = cv2.imread("temp/captcha.jpg")
            with open("temp/captcha.jpg", 'rb') as f:
                captcha_b = f.read()
            bboxes = det.detection(captcha_b)
            result = dict()
            for i in range(len(bboxes)):
                x1, y1, x2, y2 = bboxes[i]
                spec = captcha[y1:y2, x1:x2]
                cv2.imwrite(f"temp/spec_{i + 1}.jpg", spec)
                for j in range(3):
                    similarity, matched = compute_similarity(f"temp/sprite_{j + 1}.jpg", f"temp/spec_{i + 1}.jpg")
                    similarity_key = f"sprite_{j + 1}.similarity"
                    position_key = f"sprite_{j + 1}.position"
                    if similarity_key in result.keys():
                        if float(result[similarity_key]) < similarity:
                            result[similarity_key] = similarity
                            result[position_key] = f"{int((x1 + x2) / 2)},{int((y1 + y2) / 2)}"
                    else:
                        result[similarity_key] = similarity
                        result[position_key] = f"{int((x1 + x2) / 2)},{int((y1 + y2) / 2)}"
            if check_answer(result):
                for i in range(3):
                    similarity_key = f"sprite_{i + 1}.similarity"
                    position_key = f"sprite_{i + 1}.position"
                    positon = result[position_key]
                    logger.info(f"图案 {i + 1} 位于 ({positon})，匹配率：{result[similarity_key]}")
                    slideBg = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="slideBg"]')))
                    style = slideBg.get_attribute("style")
                    x, y = int(positon.split(",")[0]), int(positon.split(",")[1])
                    width_raw, height_raw = captcha.shape[1], captcha.shape[0]
                    width, height = float(get_width_from_style(style)), float(get_height_from_style(style))
                    x_offset, y_offset = float(-width / 2), float(-height / 2)
                    final_x, final_y = int(x_offset + x / width_raw * width), int(y_offset + y / height_raw * height)
                    ActionChains(driver).move_to_element_with_offset(slideBg, final_x, final_y).click().perform()
                confirm = wait.until(
                    EC.element_to_be_clickable((By.XPATH, '//*[@id="tcStatus"]/div[2]/div[2]/div/div')))
                logger.info("提交验证码")
                confirm.click()
                time.sleep(5)
                result = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="tcOperation"]')))
                if result.get_attribute("class") == 'tc-opera pointer show-success':
                    logger.info("验证码通过")
                    return
                else:
                    logger.error("验证码未通过，正在重试")
            else:
                logger.error("验证码识别失败，正在重试")
        else:
            logger.error("当前验证码识别率低，尝试刷新")
        
        # 识别失败或未通过，进行刷新重试
        driver.switch_to.default_content() # 切换到主文档
        reload = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="reload"]')))
        time.sleep(random.uniform(1, 3))
        reload.click()
        logger.info("识别失败/未通过，点击刷新按钮重试")
        time.sleep(5)
        
        # 重新切换回验证码 iframe
        wait.until(EC.frame_to_be_available_and_switch_to_it((By.ID, "tcaptcha_iframe_dy")))
        process_captcha()
        
    except TimeoutException:
        logger.error("验证码处理流程超时")
        raise
    except Exception as e:
        logger.error(f"验证码处理流程发生错误: {e}")
        raise
# --- END OF MODIFICATION ---


def download_captcha_img():
    if os.path.exists("temp"):
        for filename in os.listdir("temp"):
            file_path = os.path.join("temp", filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
    # 确保 slideBg 元素可见
    slideBg = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="slideBg"]')))
    img1_style = slideBg.get_attribute("style")
    img1_url = get_url_from_style(img1_style)
    logger.info("开始下载验证码图片(1): " + img1_url)
    download_image(img1_url, "captcha.jpg")
    sprite = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="instruction"]/div/img')))
    img2_url = sprite.get_attribute("src")
    logger.info("开始下载验证码图片(2): " + img2_url)
    download_image(img2_url, "sprite.jpg")


def check_captcha() -> bool:
    """改进的验证码检查函数"""
    try:
        raw = cv2.imread("temp/sprite.jpg")
        if raw is None:
            logger.error("无法读取验证码图片")
            return False
        
        # 图像质量检查
        h, w = raw.shape[:2]
        if h < 50 or w < 100:  # 检查图像尺寸是否合理
            logger.warning(f"验证码图片尺寸过小: {w}x{h}")
            return False
        
        # 图像清晰度检查
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian < 50:  # 低于阈值可能是模糊图像
            logger.warning(f"验证码图片清晰度不足: {laplacian}")
            return False
            
        # 分割和保存三个子图像
        for i in range(3):
            w_segment = w // 3
            # 添加一定的边界裕度，避免分割到边缘
            start_x = max(0, w_segment * i + 2)
            end_x = min(w, w_segment * (i + 1) - 2)
            temp = raw[:, start_x:end_x]
            cv2.imwrite(f"temp/sprite_{i + 1}.jpg", temp)
            
            # 图像识别检查
            with open(f"temp/sprite_{i + 1}.jpg", mode="rb") as f:
                temp_rb = f.read()
            try:
                # 使用全局 ocr 实例
                global ocr
                result = ocr.classification(temp_rb)
                if result in ["0", "1"]:
                    logger.warning(f"发现无效验证码: sprite_{i + 1}.jpg = {result}")
                    return False
            except Exception as e:
                logger.error(f"OCR识别出错: {e}")
                return False
        
        return True
    except Exception as e:
        logger.error(f"验证码检查失败: {e}")
        return False


# 检查是否存在重复坐标，快速判断识别错误
def check_answer(d: dict) -> bool:
    """改进的答案检查函数，不仅检查重复坐标，还检查相似度阈值"""
    # 检查是否有重复值
    flipped = dict()
    for key in d.keys():
        # 仅检查位置键
        if key.endswith(".position"):
            flipped[d[key]] = key
    
    # 因为 d 字典中同时包含 similarity 和 position 键，所以不能直接比较 d.values() 的长度
    # 我们应该检查 position 键的数量是否和去重后的位置数量相等
    position_keys = [k for k in d.keys() if k.endswith(".position")]
    if len(position_keys) != len(flipped.keys()):
        return False
    
    # 检查相似度是否达到最低阈值
    min_similarity_threshold = 0.3  # 设置最低相似度阈值
    for i in range(3):
        similarity_key = f"sprite_{i + 1}.similarity"
        if similarity_key in d and float(d[similarity_key]) < min_similarity_threshold:
            logger.warning(f"相似度不足: {similarity_key} = {d[similarity_key]}")
            return False
    
    # 检查位置是否合理分布（避免太集中）
    positions = []
    for i in range(3):
        position_key = f"sprite_{i + 1}.position"
        if position_key in d:
            x, y = map(int, d[position_key].split(","))
            positions.append((x, y))
    
    # 检查位置分布是否合理
    if len(positions) == 3:
        # 计算x坐标的分布范围
        x_coords = [p[0] for p in positions]
        x_range = max(x_coords) - min(x_coords)
        
        # 如果三个点太集中，可能识别有误
        if x_range < 50:  # 假设阈值为50像素
            logger.warning(f"位置分布过于集中: x范围 = {x_range}")
            return False
    
    return True


def preprocess_image(image):
    """图像预处理函数，提高特征匹配准确率"""
    # 转换为灰度图（如果不是）
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # 自适应阈值二值化，增强对比度
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # 形态学操作，增强特征
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return morph

def compute_similarity(img1_path, img2_path):
    """优化的相似度计算函数"""
    # 读取图像
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        logger.error("相似度计算：无法读取图像文件")
        return 0.0, 0
    
    # 图像尺寸标准化
    max_dim = 100
    if img1.shape[0] > max_dim or img1.shape[1] > max_dim:
        # 如果图像太大，先缩小以提高处理速度
        scale = max_dim / max(img1.shape)
        img1 = cv2.resize(img1, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    if img2.shape[0] > max_dim or img2.shape[1] > max_dim:
        scale = max_dim / max(img2.shape)
        img2 = cv2.resize(img2, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    # 图像预处理
    img1 = preprocess_image(img1)
    img2 = preprocess_image(img2)

    # 使用SIFT特征提取
    try:
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            return 0.0, 0

        # 使用FLANN匹配器，比BFMatcher更高效
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # 增加检查次数以提高准确率
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1.astype('float32'), des2.astype('float32'), k=2)

        # 应用比例测试筛选好的匹配点，使用更严格的阈值0.7
        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    good.append(m)

        if len(good) == 0:
            return 0.0, 0

        # 计算相似度时考虑特征点总数和匹配点比例
        # 同时考虑特征点数量的影响，避免小图像的匹配点少但比例高的问题
        feature_factor = min(1.0, len(kp1) / 100.0, len(kp2) / 100.0)  # 归一化特征点数量因子
        match_ratio = len(good) / min(len(des1), len(des2))  # 使用最小特征点数作为分母更合理
        
        # 综合相似度计算
        similarity = match_ratio * 0.7 + feature_factor * 0.3
        
        return similarity, len(good)
    except Exception as e:
        # 有些环境可能没有 SIFT/FLANN，例如某些 OpenCV 版本编译不完整
        logger.error(f"相似度计算出错 (可能缺少 SIFT/FLANN 模块): {e}")
        # 备用：使用模板匹配或其他简单方法 (此处省略备用逻辑，直接返回 0.0)
        return 0.0, 0


def get_current_points(driver: WebDriver, wait: WebDriverWait) -> int:
    """获取当前积分并返回，失败返回 0"""
    try:
        driver.get("https://app.rainyun.com/dashboard")
        wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
        time.sleep(2) # 等待渲染

        # 定位积分元素
        points_xpath = '//*[@id="app"]/div[1]/div[3]/div[2]/div/div/div[2]/div[1]/div[1]/div/p/div/h3'
        points_raw = wait.until(EC.visibility_of_element_located((By.XPATH, points_xpath))).get_attribute("textContent")
        current_points = int(''.join(re.findall(r'\d+', points_raw)))
        return current_points
    except Exception as e:
        logger.error(f"获取当前积分失败: {e}")
        return 0


# --- START OF MODIFICATION: 改进 sign_in_account 逻辑，使用积分对比判断签到成功 ---
def sign_in_account(user, pwd, debug=False, headless=False):
    """
    单个账户登录签到函数
    
    Args:
        user: 用户名
        pwd: 密码
        debug: 是否开启调试模式
        headless: 是否使用无头模式
        
    Returns:
        tuple: (成功状态, 用户名, 积分信息, 错误信息)
    """
    # 连接超时等待
    timeout = 15
    driver = None
    
    try:
        logger.info(f"开始处理账户: {user}")
        
        if not debug:
            delay_sec = random.randint(5, 10)
            logger.info(f"随机延时等待 {delay_sec} 秒")
            time.sleep(delay_sec)
        
        # 确保 ocr 和 det 是全局变量，供 process_captcha/check_captcha 使用
        global ocr, det, wait
        
        logger.info("初始化 ddddocr")
        ocr = ddddocr.DdddOcr(ocr=True, show_ad=False)
        det = ddddocr.DdddOcr(det=True, show_ad=False)
        
        logger.info("初始化 Selenium")
        # 传递 headless 参数给 init_selenium
        driver = init_selenium(debug=debug, headless=headless)
        
        # 过 Selenium 检测
        with open("stealth.min.js", mode="r") as f:
            js = f.read()
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": js
        })
        
        logger.info("发起登录请求")
        driver.get("https://app.rainyun.com/auth/login")
        wait = WebDriverWait(driver, timeout) # 设置全局 wait 实例
        
        # 改进的登录逻辑，添加重试机制
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 使用更可靠的定位方式
                username = wait.until(EC.visibility_of_element_located((By.NAME, 'login-field')))
                password = wait.until(EC.visibility_of_element_located((By.NAME, 'login-password')))
                
                # 尝试多种方式定位登录按钮
                try:
                    login_button = wait.until(EC.element_to_be_clickable((By.XPATH,
                                                                        '//*[@id="app"]/div[1]/div[1]/div/div[2]/fade/div/div/span/form/button')))
                except:
                    try:
                        login_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[type="submit"]')))
                    except:
                        login_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//button[contains(text(), "登录")]')))
                
                # 清除可能存在的输入
                username.clear()
                password.clear()
                
                # 添加输入延迟，模拟真实用户
                username.send_keys(user)
                time.sleep(0.5)
                password.send_keys(pwd)
                time.sleep(0.5)
                
                # 使用JavaScript点击，避免元素遮挡问题
                driver.execute_script("arguments[0].click();", login_button)
                logger.info(f"登录尝试 {retry_count + 1}/{max_retries}")
                break # 登录点击操作完成，跳出重试循环
            except TimeoutException:
                retry_count += 1
                if retry_count < max_retries:
                    logger.warning(f"登录页面加载失败，{retry_count}秒后重试...")
                    time.sleep(retry_count)
                    driver.refresh()
                else:
                    logger.error("页面加载超时，请尝试延长超时时间或切换到国内网络环境！")
                    raise Exception("登录页面加载超时或失败。")
        
        try:
            # 等待验证码 iframe 出现并切换
            login_captcha = wait.until(EC.visibility_of_element_located((By.ID, 'tcaptcha_iframe_dy')))
            logger.warning("触发登录验证码！")
            driver.switch_to.frame("tcaptcha_iframe_dy")
            process_captcha()
        except TimeoutException:
            logger.info("未触发登录验证码")
        
        time.sleep(5)
        driver.switch_to.default_content()
        
        # 验证登录状态并处理赚取积分
        if "dashboard" in driver.current_url:
            logger.info("登录成功！")
            
            # --- 1. 获取签到前积分 ---
            initial_points = get_current_points(driver, wait)
            if initial_points > 0:
                logger.info(f"签到前积分: {initial_points} | 约为 {initial_points / 2000:.2f} 元")
            else:
                logger.warning("未能获取签到前积分，将无法精确判断签到是否成功。")

            logger.info("正在转到赚取积分页")
            driver.get("https://app.rainyun.com/account/reward/earn")

            # --- 2. 尝试点击赚取积分按钮 ---
            max_click_retries = 3
            for _ in range(max_click_retries):
                try:
                    logger.info("等待赚取积分页面加载...")
                    wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
                    time.sleep(3) # 额外等待确保页面完全渲染
                    
                    # 使用多种策略查找赚取积分按钮
                    earn = None
                    strategies = [
                        (By.XPATH, '//a[contains(@href, "earn") and contains(text(), "赚取")]'),
                        (By.CSS_SELECTOR, 'a[href*="earn"]'),
                        (By.XPATH, '//*[@id="app"]/div[1]/div[3]/div[2]/div/div/div[2]/div[2]/div/div/div/div[1]/div/div[1]/div/div[1]/div/span[2]/a')
                    ]
                    
                    for by, selector in strategies:
                        try:
                            earn = wait.until(EC.element_to_be_clickable((by, selector)))
                            logger.info(f"使用策略 {by}={selector} 找到赚取积分按钮")
                            break
                        except Exception as e:
                            logger.debug(f"策略 {by}={selector} 未找到按钮或不可点击。")
                            continue
                    
                    if earn:
                        # 滚动到元素位置并点击
                        driver.execute_script("arguments[0].scrollIntoView(true);", earn)
                        time.sleep(1)
                        logger.info("点击赚取积分按钮")
                        driver.execute_script("arguments[0].click();", earn)
                        
                        # 处理可能出现的二次验证码
                        try:
                            logger.info("检查是否需要二次验证码")
                            wait.until(EC.frame_to_be_available_and_switch_to_it((By.ID, "tcaptcha_iframe_dy")))
                            logger.info("处理二次验证码")
                            process_captcha() 
                            driver.switch_to.default_content()
                        except TimeoutException:
                            logger.info("未触发二次验证码或验证码框架加载失败")
                            driver.switch_to.default_content()
                        
                        logger.info("赚取积分操作完成，等待积分刷新...")
                        time.sleep(5) # 留出时间让服务器处理和页面刷新
                        break
                    else:
                        logger.warning("未找到赚取积分按钮，刷新页面重试...")
                        driver.refresh()
                        time.sleep(3)
                except Exception as e:
                    logger.error(f"尝试点击赚取积分按钮时出错: {e}")
                    time.sleep(3)
            else:
                logger.error("多次尝试后仍无法点击赚取积分按钮。")
            
            # --- 3. 检查签到后积分并判断结果 ---
            current_points = get_current_points(driver, wait)
            
            if initial_points > 0 and current_points > initial_points:
                added_points = current_points - initial_points
                logger.info(f"任务执行成功！积分已增加 {added_points} 点。")
                logger.info(f"当前剩余积分: {current_points} | 约为 {current_points / 2000:.2f} 元")
                return True, user, current_points, None
            elif current_points > 0 and (initial_points > 0 and current_points == initial_points):
                logger.warning(f"当前剩余积分: {current_points} (与签到前积分相同: {initial_points})")
                logger.info("任务执行完毕，但积分未增加，可能已签到或点击失败。")
                # 流程已执行，标记为成功，但给出警告信息
                return True, user, current_points, "积分未增加，可能已签到或点击失败。"
            else:
                # 可能是 initial_points 为 0，无法比较，但 current_points 成功获取，默认视为成功
                logger.info(f"当前剩余积分: {current_points} | 约为 {current_points / 2000:.2f} 元")
                logger.info("任务执行成功！")
                return True, user, current_points, None
        else:
            logger.error("登录失败！")
            
            return False, user, 0, "登录失败，未能进入仪表盘页面，请检查账号密码或验证码处理逻辑。"

    except Exception as e: # 【新增】捕获未处理的异常
        err_msg = f"脚本运行期间发生致命异常: {str(e)}"
        logger.error(err_msg, exc_info=True)
        
        return False, user, 0, err_msg

    finally: # 【新增】确保浏览器关闭
        if driver:
            logger.info("正在关闭浏览器...")
            try:
                driver.quit()
            except:
                pass
# --- END OF MODIFICATION ---

# 修改main函数，移除重复的变量定义
# 修改随机延时等待设置
if __name__ == "__main__":
    # 环境变量判断是否在GitHub Actions中运行
    is_github_actions = os.environ.get("GITHUB_ACTIONS", "false") == "true"
    # 从环境变量读取模式设置
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    headless = os.environ.get('HEADLESS', 'false').lower() == 'true'
    
    # 如果在GitHub Actions环境中，强制使用无头模式
    if is_github_actions:
        headless = True
    
    # 以下代码保持不变...
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    # 定义全局 ocr/det/wait 变量以防止 process_captcha 报错
    ocr = None
    det = None
    wait = None

    ver = "2.3" # 版本号更新
    logger.info("------------------------------------------------------------------")
    logger.info(f"雨云自动签到工作流 v{ver} by 筱序二十 ~")
    logger.info("推广链接https://www.rainyun.com/bv_?s=rqd")
    logger.info("支持我https://rewards.qxzhan.cn/")
    logger.info("Github发布页: https://github.com/scfcn/Rainyun-Qiandao")
    logger.info("------------------------------------------------------------------")
    
    # 读取账户信息
    accounts = []
    
    # 读取环境变量
    users_env = os.environ.get("RAINYUN_USER", "")
    passwords_env = os.environ.get("RAINYUN_PASS", "")
    
    # 按行分割，过滤空行
    users = [user.strip() for user in users_env.split('\n') if user.strip()]
    passwords = [pwd.strip() for pwd in passwords_env.split('\n') if pwd.strip()]
    
    # 检查用户和密码数量是否匹配
    if len(users) == len(passwords):
        if len(users) > 0:
            logger.info(f"读取到 {len(users)} 个账户配置")
            for user, pwd in zip(users, passwords):
                accounts.append((user, pwd))
        else:
            logger.warning("未读取到有效账户配置")
    else:
        logger.error(f"用户和密码数量不匹配: {len(users)} 个用户, {len(passwords)} 个密码")
        exit(1)
    
    # 确保有账户配置
    if not accounts:
        err_msg = "错误: 未设置用户名或密码，请在环境变量中设置RAINYUN_USER和RAINYUN_PASS（支持多行格式，每行一个用户名/密码，数量需匹配）"
        print(err_msg)
        # --- START OF MODIFICATION: 添加配置错误通知 ---
        try:
            send("雨云签到配置错误", err_msg)
        except Exception as e:
            print(f"发送通知失败: {e}")
        # --- END OF MODIFICATION ---
        exit(1)
    
    logger.info(f"共读取到 {len(accounts)} 个账户")
    
    # 收集所有账户的签到结果
    results = []
    
    # 遍历所有账户，依次执行签到
    for i, (user, pwd) in enumerate(accounts, 1):
        logger.info(f"\n=== 开始处理第 {i} 个账户: {user} ===")
        # 传入 driver/wait 等变量
        result = sign_in_account(user, pwd, debug=debug, headless=headless)
        results.append(result)
        logger.info(f"=== 第 {i} 个账户处理完成 ===\n")
    
    logger.info("所有账户处理完成")
    
    # 生成统一通知
    success_count = sum(1 for r in results if r[0])
    total_count = len(results)
    
    if success_count == total_count:
        notification_title = f"✅ 雨云自动签到完成 - 全部成功"
    elif success_count > 0:
        notification_title = f"⚠️ 雨云自动签到完成 - 部分成功 ({success_count}/{total_count})"
    else:
        notification_title = f"❌ 雨云自动签到完成 - 全部失败"
    
    notification_content = f"雨云自动签到结果汇总：\n\n总账户数: {total_count}\n成功账户数: {success_count}\n失败账户数: {total_count - success_count}\n\n详细结果：\n"
    
    for i, (success, user, points, error_msg) in enumerate(results, 1):
        if success:
            # 检查是否有警告信息
            if error_msg and "积分未增加" in error_msg:
                notification_content += f"{i}. ⚠️ {user}\n   积分: {points} | 约 {points / 2000:.2f} 元 (已签到)\n"
            else:
                notification_content += f"{i}. ✅ {user}\n   积分: {points} | 约 {points / 2000:.2f} 元\n"
        else:
            notification_content += f"{i}. ❌ {user}\n   错误: {error_msg}\n"
    
    # 发送统一通知
    try:
        send(notification_title, notification_content)
        logger.info("统一通知发送成功")
    except Exception as e:
        logger.error(f"发送通知失败: {e}")
