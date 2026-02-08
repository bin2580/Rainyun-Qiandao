import logging
import os
import random
import re
import time
import platform  # 新增：自动识别系统

import cv2
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

# 注意：ICR 库需要确保已安装，若未安装请执行：pip install ICR
import ICR


def init_selenium() -> WebDriver:
    ops = Options()
    # 通用配置
    ops.add_argument("--no-sandbox")  # 解决Linux沙箱问题
    ops.add_argument("--disable-dev-shm-usage")  # 解决内存不足问题
    ops.add_argument("--disable-blink-features=AutomationControlled")  # 绕过检测
    ops.add_experimental_option("excludeSwitches", ["enable-automation"])
    ops.add_experimental_option('useAutomationExtension', False)
    
    # 自动识别系统（替代手动设置linux变量）
    system = platform.system().lower()
    if debug:
        ops.add_experimental_option("detach", True)
    if system == "linux":
        # Linux/青龙面板专属配置
        ops.add_argument("--headless=new")  # 新版无头模式（比旧版--headless更稳定）
        ops.add_argument("--disable-gpu")
        # 使用系统自带的chromedriver（无需手动放置文件）
        try:
            # 优先使用系统路径的chromedriver
            driver = webdriver.Chrome(service=Service("/usr/bin/chromedriver"), options=ops)
        except:
            # 备用：自动查找chromedriver
            driver = webdriver.Chrome(options=ops)
    else:
        # Windows系统配置
        driver = webdriver.Chrome(service=Service("chromedriver.exe"), options=ops)
    
    return driver


def download_image(url, filename):
    os.makedirs("temp", exist_ok=True)
    # 增加请求头，避免被反爬
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # 抛出HTTP错误
        path = os.path.join("temp", filename)
        with open(path, "wb") as f:
            f.write(response.content)
        return True
    except Exception as e:
        logger.error(f"下载图片失败：{e}")
        return False


def get_url_from_style(style):
    """从style属性中提取图片URL"""
    match = re.search(r'url\(["\']?(.*?)["\']?\)', style)
    if match:
        return match.group(1)
    logger.error("未从style中提取到URL")
    return ""


def get_width_from_style(style):
    """从style属性中提取宽度"""
    match = re.search(r'width:\s*([\d.]+)px', style)
    if match:
        return match.group(1)
    logger.error("未从style中提取到宽度")
    return "0"


def get_height_from_style(style):
    """从style属性中提取高度"""
    match = re.search(r'height:\s*([\d.]+)px', style)
    if match:
        return match.group(1)
    logger.error("未从style中提取到高度")
    return "0"


def process_captcha():
    """处理验证码识别与点击"""
    try:
        download_captcha_img()
        logger.info("开始识别验证码")
        captcha_path = "temp/captcha.jpg"
        sprite_path = "temp/sprite.jpg"
        # 检查图片是否存在
        if not os.path.exists(captcha_path) or not os.path.exists(sprite_path):
            logger.error("验证码图片缺失，无法识别")
            raise Exception("验证码图片缺失")
        
        captcha = cv2.imread(captcha_path)
        if captcha is None:
            logger.error("验证码图片读取失败")
            raise Exception("验证码图片读取失败")
        
        result = ICR.main(captcha_path, sprite_path)
        if not result:
            logger.error("ICR识别验证码返回空结果")
            raise Exception("验证码识别失败")
        
        for info in result:
            rect = info['bg_rect']
            x, y = int(rect[0] + (rect[2] / 2)), int(rect[1] + (rect[3] / 2))
            logger.info(f"图案 {info['sprite_idx'] + 1} 位于 ({x}, {y})")
            slideBg = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="slideBg"]')))
            style = slideBg.get_attribute("style")
            width_raw, height_raw = captcha.shape[1], captcha.shape[0]
            width, height = float(get_width_from_style(style)), float(get_height_from_style(style))
            x_offset, y_offset = float(-width / 2), float(-height / 2)
            final_x, final_y = int(x_offset + x / width_raw * width), int(y_offset + y / height_raw * height)
            # 模拟人类点击（增加小随机偏移）
            final_x += random.randint(-2, 2)
            final_y += random.randint(-2, 2)
            ActionChains(driver).move_to_element_with_offset(slideBg, final_x, final_y).click().perform()
            time.sleep(random.uniform(0.5, 1.5))  # 随机延时，模拟人类操作
        
        # 点击确认按钮
        confirm = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="tcStatus"]/div[2]/div[2]/div/div')))
        logger.info("提交验证码")
        confirm.click()
        time.sleep(5)
        
        # 验证验证码是否通过
        result = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="tcOperation"]')))
        if result.get_attribute("class") == 'tc-opera pointer show-success':
            logger.info("验证码通过")
            return
        else:
            logger.error("验证码未通过，正在重试")
            reload = driver.find_element(By.XPATH, '//*[@id="reload"]')
            time.sleep(5)
            reload.click()
            time.sleep(5)
            process_captcha()  # 递归重试
    except TimeoutException:
        logger.error("获取验证码元素超时")
    except Exception as e:
        logger.error(f"处理验证码异常：{e}")
        # 重试逻辑
        try:
            reload = driver.find_element(By.XPATH, '//*[@id="reload"]')
            reload.click()
            time.sleep(5)
            process_captcha()
        except:
            logger.error("验证码重试失败")


def download_captcha_img():
    """下载验证码背景图和滑块图"""
    # 清空temp目录
    if os.path.exists("temp"):
        for filename in os.listdir("temp"):
            file_path = os.path.join("temp", filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"删除临时文件失败：{e}")
    
    # 下载背景图
    try:
        slideBg = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="slideBg"]')))
        img1_style = slideBg.get_attribute("style")
        img1_url = get_url_from_style(img1_style)
        if img1_url:
            logger.info("开始下载验证码图片(1): " + img1_url)
            download_image(img1_url, "captcha.jpg")
    except Exception as e:
        logger.error(f"下载验证码背景图失败：{e}")
    
    # 下载滑块图
    try:
        sprite = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="instruction"]/div/img')))
        img2_url = sprite.get_attribute("src")
        if img2_url:
            logger.info("开始下载验证码图片(2): " + img2_url)
            download_image(img2_url, "sprite.jpg")
    except Exception as e:
        logger.error(f"下载验证码滑块图失败：{e}")


if __name__ == "__main__":
    # 配置项
    timeout = 20  # 延长超时时间（适配网络波动）
    max_delay = 0  # 最大随机等待分钟数
    user = "qwer2580"  # 替换为你的雨云用户名
    pwd = "qwer2580"  # 替换为你的雨云密码
    debug = False  # 调试模式（True时浏览器不自动关闭）
    # linux = False  # 已改为自动识别系统，无需手动设置

    # 初始化日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    
    # 版本信息
    ver = "2.3"
    logger.info("------------------------------------------------------------------")
    logger.info(f"雨云签到工具 v{ver} by SerendipityR ~")
    logger.info("Github发布页: https://github.com/SerendipityR-2022/Rainyun-Qiandao")
    logger.info("------------------------------------------------------------------")
    
    # 随机延时
    delay = random.randint(0, max_delay)
    delay_sec = random.randint(0, 6)
    if not debug:
        logger.info(f"随机延时等待 {delay} 分钟 {delay_sec} 秒")
        time.sleep(delay * 60 + delay_sec)
    
    # 初始化Selenium
    logger.info("初始化 Selenium")
    try:
        driver = init_selenium()
    except Exception as e:
        logger.error(f"Selenium初始化失败：{e}")
        logger.error("请检查ChromeDriver是否安装正确，或执行：apt install chromium chromium-driver")
        exit(1)
    
    # 绕过Selenium检测（需要确保stealth.min.js文件存在）
    try:
        with open("stealth.min.js", mode="r", encoding="utf-8") as f:
            js = f.read()
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": js})
    except FileNotFoundError:
        logger.warning("未找到stealth.min.js文件，可能无法绕过Selenium检测")
    except Exception as e:
        logger.warning(f"加载反检测脚本失败：{e}")
    
    # 登录流程
    logger.info("发起登录请求")
    driver.get("https://app.rainyun.com/auth/login")
    wait = WebDriverWait(driver, timeout)
    
    try:
        # 输入账号密码
        username = wait.until(EC.visibility_of_element_located((By.NAME, 'login-field')))
        password = wait.until(EC.visibility_of_element_located((By.NAME, 'login-password')))
        login_button = wait.until(EC.element_to_be_clickable((By.XPATH,
            '//*[@id="app"]/div[1]/div[1]/div/div[2]/fade/div/div/span/form/button')))
        
        username.send_keys(user)
        password.send_keys(pwd)
        time.sleep(random.uniform(0.5, 1.5))  # 模拟人类输入延时
        login_button.click()
    except TimeoutException:
        logger.error("页面加载超时，请检查网络或延长timeout值！")
        driver.quit()
        exit(1)
    
    # 处理登录验证码
    try:
        login_captcha = wait.until(EC.visibility_of_element_located((By.ID, 'tcaptcha_iframe_dy')))
        logger.warning("触发登录验证码！")
        driver.switch_to.frame("tcaptcha_iframe_dy")
        process_captcha()
    except TimeoutException:
        logger.info("未触发登录验证码")
    except Exception as e:
        logger.error(f"处理登录验证码异常：{e}")
    
    # 回到主页面
    time.sleep(5)
    driver.switch_to.default_content()
    
    # 验证登录是否成功
    if driver.current_url == "https://app.rainyun.com/dashboard":
        logger.info("登录成功！")
        logger.info("正在转到赚取积分页")
        driver.get("https://app.rainyun.com/account/reward/earn")
        driver.implicitly_wait(5)
        
        # 点击赚取积分
        try:
            earn = wait.until(EC.element_to_be_clickable((By.XPATH,
                '//*[@id="app"]/div[1]/div[3]/div[2]/div/div/div[2]/div[2]/div/div/div/div[1]/div/div[1]/div/div[1]/div/span[2]/a')))
            logger.info("点击赚取积分")
            earn.click()
            time.sleep(2)
            
            # 处理赚取积分的验证码
            try:
                driver.switch_to.frame("tcaptcha_iframe_dy")
                process_captcha()
            except TimeoutException:
                logger.info("赚取积分未触发验证码")
            except Exception as e:
                logger.error(f"处理赚取积分验证码异常：{e}")
            
            # 回到主页面，获取积分
            driver.switch_to.default_content()
            driver.implicitly_wait(5)
            try:
                points_raw = driver.find_element(By.XPATH,
                    '//*[@id="app"]/div[1]/div[3]/div[2]/div/div/div[2]/div[1]/div[1]/div/p/div/h3').text
                current_points = int(''.join(re.findall(r'\d+', points_raw)))
                logger.info(f"当前剩余积分: {current_points} | 约为 {current_points / 2000:.2f} 元")
            except Exception as e:
                logger.error(f"获取积分失败：{e}")
            
            logger.info("任务执行成功！")
        except Exception as e:
            logger.error(f"赚取积分操作失败：{e}")
    else:
        logger.error("登录失败！请检查账号密码或验证码处理逻辑")
    
    # 关闭浏览器
    if not debug:
        time.sleep(3)
        driver.quit()
    logger.info("程序结束")
