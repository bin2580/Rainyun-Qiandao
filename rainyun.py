import logging
import os
import random
import re
import stat
import time

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

import ICR

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from webdriver_manager.chrome import ChromeDriverManager
    try:
        from webdriver_manager.core.utils import ChromeType
    except ImportError:
        try:
            from webdriver_manager.chrome import ChromeType
        except ImportError:
            ChromeType = None
except ImportError:
    print("webdriver_manager未安装，将使用备用方式")
    ChromeDriverManager = None
    ChromeType = None

try:
    from notify import send
    print("已加载通知模块 (notify.py)")
except ImportError:
    print("警告: 未找到 notify.py，将无法发送通知。")
    def send(*args, **kwargs):
        pass

AD_URL = os.environ.get("AD_URL", "https://scfc.dpdns.org/ad.json")

AD_TEXT = None
AD_LINK = None
AD_ENABLED = True
AD_LIST = None
MIN_VERSION = None
LATEST_VERSION = None
UPDATE_URL = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_remote_ad():
    global AD_TEXT, AD_LINK, AD_ENABLED, AD_LIST, MIN_VERSION, LATEST_VERSION, UPDATE_URL
    if not AD_URL:
        return
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        response = requests.get(AD_URL, timeout=5, proxies={"http": None, "https": None}, verify=False)
        if response.status_code == 200:
            import json
            ad_data = json.loads(response.text)
            
            if "enabled" in ad_data:
                AD_ENABLED = ad_data["enabled"]
            
            if "min_version" in ad_data:
                MIN_VERSION = ad_data["min_version"]
            
            if "latest_version" in ad_data:
                LATEST_VERSION = ad_data["latest_version"]
            
            if "update_url" in ad_data:
                UPDATE_URL = ad_data["update_url"].strip().strip('`').strip("'").strip('"')
            
            if "ads" in ad_data and isinstance(ad_data["ads"], list):
                AD_LIST = ad_data["ads"]
            else:
                if "text" in ad_data:
                    AD_TEXT = ad_data["text"]
                if "link" in ad_data:
                    AD_LINK = ad_data["link"]
        else:
            print(f"⚠️ 远程获取失败，状态码: {response.status_code}")
    except Exception as e:
        print(f"⚠️ 远程获取异常: {e}")


fetch_remote_ad()


def auto_update(current_ver):
    global LATEST_VERSION, UPDATE_URL
    if not LATEST_VERSION or LATEST_VERSION == current_ver:
        return
    
    print(f"🔄 开始自动更新到 v{LATEST_VERSION}...")
    
    try:
        import subprocess
        import shutil
        
        git_path = shutil.which('git')
        if not git_path:
            print("⚠️ 未找到 git，尝试直接下载...")
            download_update()
            return
        
        print(f"📥 正在使用 git 同步最新版本...")
        
        subprocess.run(['git', 'fetch', '--all'], check=True, capture_output=True, text=True)
        subprocess.run(['git', 'reset', '--hard', 'origin/main'], check=True, capture_output=True, text=True)
        
        print(f"✅ 更新完成！已同步到 v{LATEST_VERSION}")
        print(f"📝 请重新运行脚本以使用新版本")
        exit(0)
    except subprocess.CalledProcessError as e:
        print(f"❌ git 同步失败: {e}")
        print(f"📥 尝试直接下载...")
        download_update()
    except Exception as e:
        print(f"❌ 自动更新失败: {e}")
        if UPDATE_URL:
            print(f"📥 请手动更新: {UPDATE_URL}")


def download_update():
    global LATEST_VERSION, UPDATE_URL
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        repo_url = "https://github.com/scfcn/Rainyun-Qiandao"
        raw_url = f"{repo_url}/raw/main/rainyun.py"
        
        print(f"📥 正在下载最新版本...")
        response = requests.get(raw_url, timeout=30, proxies={"http": None, "https": None}, verify=False)
        
        if response.status_code == 200:
            new_content = response.text
            
            with open(__file__, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"✅ 更新完成！已更新到 v{LATEST_VERSION}")
            print(f"📝 请重新运行脚本以使用新版本")
            exit(0)
        else:
            print(f"❌ 下载失败，状态码: {response.status_code}")
            if UPDATE_URL:
                print(f"📥 请手动更新: {UPDATE_URL}")
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        if UPDATE_URL:
            print(f"📥 请手动更新: {UPDATE_URL}")


def init_selenium(debug=False, headless=False) -> WebDriver:
    ops = Options()
    if headless or os.environ.get("GITHUB_ACTIONS", "false") == "true":
        for option in ['--headless', '--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu']:
            ops.add_argument(option)
    ops.add_argument('--window-size=1920,1080')
    ops.add_argument('--disable-blink-features=AutomationControlled')
    ops.add_argument('--no-proxy-server')
    ops.add_argument('--lang=zh-CN')
    
    is_github_actions = os.environ.get("GITHUB_ACTIONS", "false") == "true"
    if debug and not is_github_actions:
        ops.add_experimental_option("detach", True)
    
    try:
        if ChromeDriverManager:
            if ChromeType and hasattr(ChromeType, 'GOOGLE'):
                manager = ChromeDriverManager(chrome_type=ChromeType.GOOGLE)
            else:
                manager = ChromeDriverManager()
            driver_path = manager.install()
            
            if os.path.isfile(driver_path):
                filename = os.path.basename(driver_path)
                if 'chromedriver' in filename.lower() and 'THIRD_PARTY_NOTICES' not in filename:
                    os.chmod(driver_path, os.stat(driver_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                    service = Service(driver_path)
                    driver = webdriver.Chrome(service=service, options=ops)
                    return driver
            
            if os.path.isdir(driver_path):
                driver_dir = driver_path
            else:
                driver_dir = os.path.dirname(driver_path)
            
            chromedriver_path = os.path.join(driver_dir, 'chromedriver')
            if os.path.isfile(chromedriver_path):
                os.chmod(chromedriver_path, os.stat(chromedriver_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                service = Service(chromedriver_path)
                driver = webdriver.Chrome(service=service, options=ops)
                return driver
            
            for root, dirs, files in os.walk(driver_dir):
                if root.endswith('chromedriver-linux64'):
                    for file in files:
                        if file == 'chromedriver':
                            correct_path = os.path.join(root, file)
                            os.chmod(correct_path, os.stat(correct_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                            service = Service(correct_path)
                            driver = webdriver.Chrome(service=service, options=ops)
                            return driver
    except Exception as e:
        print(f"webdriver-manager失败: {e}")

    try:
        driver = webdriver.Chrome(options=ops)
        return driver
    except Exception:
        pass
        
    raise Exception("无法初始化Selenium WebDriver")


def download_image(url, filename):
    os.makedirs("temp", exist_ok=True)
    try:
        response = requests.get(url, timeout=10, proxies={"http": None, "https": None}, verify=False)
        if response.status_code == 200:
            with open(os.path.join("temp", filename), "wb") as f:
                f.write(response.content)
            return True
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


def process_captcha(driver, wait):
    try:
        download_captcha_img(driver, wait)
        logger.info("开始识别验证码")
        captcha = cv2.imread("temp/captcha.jpg")
        result = ICR.main("temp/captcha.jpg", "temp/sprite.jpg")
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
        reload = driver.find_element(By.XPATH, '//*[@id="reload"]')
        time.sleep(5)
        reload.click()
        time.sleep(5)
        process_captcha(driver, wait)
    except TimeoutException:
        logger.error("获取验证码图片失败")


def download_captcha_img(driver, wait):
    if os.path.exists("temp"):
        for filename in os.listdir("temp"):
            file_path = os.path.join("temp", filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
    slideBg = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="slideBg"]')))
    img1_style = slideBg.get_attribute("style")
    img1_url = get_url_from_style(img1_style)
    logger.info("开始下载验证码图片(1): " + img1_url)
    download_image(img1_url, "captcha.jpg")
    sprite = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="instruction"]/div/img')))
    img2_url = sprite.get_attribute("src")
    logger.info("开始下载验证码图片(2): " + img2_url)
    download_image(img2_url, "sprite.jpg")


def sign_in_account(user, pwd, debug=False, headless=False):
    timeout = 15
    driver = None
    
    try:
        if not debug:
            time.sleep(random.randint(5, 10))
        
        driver = init_selenium(debug=debug, headless=headless)
        
        try:
            with open("stealth.min.js", mode="r") as f: js = f.read()
            driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": js})
        except: pass
        
        driver.get("https://app.rainyun.com/auth/login")
        wait = WebDriverWait(driver, timeout)
        
        username = wait.until(EC.visibility_of_element_located((By.NAME, 'login-field')))
        password = wait.until(EC.visibility_of_element_located((By.NAME, 'login-password')))
        try:
            login_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="app"]/div[1]/div[1]/div/div[2]/fade/div/div/span/form/button')))
        except:
            login_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[type="submit"]')))
            
        username.clear()
        password.clear()
        username.send_keys(user)
        logger.info(f"已输入用户名: {user}")
        time.sleep(0.5)
        password.send_keys(pwd)
        logger.info("已输入密码")
        time.sleep(0.5)
        driver.execute_script("arguments[0].click();", login_button)
        logger.info("已点击登录按钮")
        
        try:
            wait.until(EC.visibility_of_element_located((By.ID, 'tcaptcha_iframe_dy')))
            logger.warning("触发验证码")
            driver.switch_to.frame("tcaptcha_iframe_dy")
            process_captcha(driver, wait)
        except TimeoutException:
            logger.info("未触发验证码")
            pass
        
        time.sleep(5)
        driver.switch_to.default_content()
        
        current_url = driver.current_url
        logger.info(f"当前URL: {current_url}")
        
        if "dashboard" in current_url or ("app.rainyun.com" in current_url and "login" not in current_url):
            logger.info("登录成功")
            
            for _ in range(3):
                try:
                    driver.get("https://app.rainyun.com/account/reward/earn")
                    wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
                    time.sleep(3)

                    try:
                        claim_btns = driver.find_elements(By.XPATH, "//span[contains(text(),'每日签到')]/following::a[contains(@href,'/account/reward/earn')][1]")
                        if any(el.is_displayed() for el in claim_btns):
                            logger.info("开始签到")
                        else:
                            completed = driver.find_elements(By.XPATH, "//span[contains(text(),'每日签到')]/following::span[contains(text(),'已完成')][1]")
                            if any(el.is_displayed() for el in completed):
                                logger.info("今日已签到")
                                try:
                                    points_raw = driver.find_element(By.XPATH, '//*[@id="app"]/div[1]/div[3]/div[2]/div/div/div[2]/div[1]/div[1]/div/p/div/h3').get_attribute("textContent")
                                    current_points = int(''.join(re.findall(r'\d+', points_raw)))
                                except:
                                    current_points = 0
                                return True, user, current_points, None
                    except Exception:
                        pass

                    strategies = [
                        (By.XPATH, '//*[@id="app"]/div[1]/div[3]/div[2]/div/div/div[2]/div[2]/div/div/div/div[1]/div/div[1]/div/div[1]/div/span[2]/a'),
                        (By.XPATH, '//a[contains(@href, "earn") and contains(text(), "赚取")]'),
                        (By.CSS_SELECTOR, 'a[href*="earn"]')
                    ]
                    
                    earn = None
                    for by, selector in strategies:
                        try:
                            earn = wait.until(EC.element_to_be_clickable((by, selector)))
                            break
                        except: continue
                    
                    if earn:
                        driver.execute_script("arguments[0].scrollIntoView(true);", earn)
                        time.sleep(1)
                        driver.execute_script("arguments[0].click();", earn)
                        
                        try:
                            WebDriverWait(driver, 15, poll_frequency=0.25).until(
                                EC.visibility_of_element_located((By.ID, "tcaptcha_iframe_dy"))
                            )
                            wait.until(EC.frame_to_be_available_and_switch_to_it((By.ID, "tcaptcha_iframe_dy")))
                            process_captcha(driver, wait)
                            driver.switch_to.default_content()
                        except TimeoutException:
                            driver.switch_to.default_content()
                        except Exception as e:
                            logger.error(f"验证码错误: {e}")
                            driver.switch_to.default_content()
                        
                        logger.info("签到完成")
                        break
                    else:
                        driver.refresh()
                        time.sleep(3)
                except Exception as e:
                    logger.error(f"出错: {e}")
                    time.sleep(3)
            
            driver.implicitly_wait(5)
            try:
                points_raw = driver.find_element(By.XPATH, '//*[@id="app"]/div[1]/div[3]/div[2]/div/div/div[2]/div[1]/div[1]/div/p/div/h3').get_attribute("textContent")
                current_points = int(''.join(re.findall(r'\d+', points_raw)))
                logger.info(f"积分: {current_points} ({current_points / 2000:.2f}元)")
            except:
                current_points = 0
                
            return True, user, current_points, None
        else:
            logger.error("登录失败")
            return False, user, 0, "登录失败"

    except Exception as e:
        logger.error(f"异常: {str(e)}", exc_info=True)
        return False, user, 0, str(e)
    finally:
        if driver:
            try: driver.quit()
            except: pass


if __name__ == "__main__":
    is_github_actions = os.environ.get("GITHUB_ACTIONS", "false") == "true"
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    headless = os.environ.get('HEADLESS', 'false').lower() == 'true'
    auto_update_enabled = os.environ.get('AUTO_UPDATE', 'true').lower() == 'true'
    if is_github_actions: headless = True
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    ver = "2.4"
    print(f"\n{'='*60}")
    print(f"  🌧️  雨云自动签到工作流 v{ver}")
    print(f"{'='*60}\n")
    
    if not AD_ENABLED:
        print("⚠️  脚本已被远程禁用，请检查配置或联系管理员")
        exit(1)
    
    if MIN_VERSION:
        try:
            from packaging import version
            current_ver = version.parse(ver)
            min_ver = version.parse(MIN_VERSION)
            if current_ver < min_ver:
                print(f"❌ 脚本版本过低！当前: {ver}, 最低要求: {MIN_VERSION}")
                if UPDATE_URL:
                    print(f"📥 更新地址: {UPDATE_URL}")
                exit(1)
        except ImportError:
            try:
                current_parts = [int(x) for x in ver.split('.')]
                min_parts = [int(x) for x in MIN_VERSION.split('.')]
                if current_parts < min_parts:
                    print(f"❌ 脚本版本过低！当前: {ver}, 最低要求: {MIN_VERSION}")
                    if UPDATE_URL:
                        print(f"📥 更新地址: {UPDATE_URL}")
                    exit(1)
            except Exception:
                pass
    
    if LATEST_VERSION and LATEST_VERSION != ver:
        print(f"📌 发现新版本: {LATEST_VERSION} (当前: {ver})")
        if UPDATE_URL:
            print(f"📥 更新地址: {UPDATE_URL}")
        print()
        if auto_update_enabled:
            auto_update(ver)
    
    if AD_LIST:
        print(f"{'─'*60}")
        for ad in AD_LIST:
            print(f"📢 {ad.get('text', '')}")
            link = ad.get('link')
            if link and link != 'null':
                print(f"🔗 {link}")
        print(f"{'─'*60}\n")
    elif AD_TEXT:
        print(f"{'─'*60}")
        print(f"📢 {AD_TEXT}")
        if AD_LINK:
            print(f"🔗 {AD_LINK}")
        print(f"{'─'*60}\n")
    
    accounts = []
    users_env = os.environ.get("RAINYUN_USER", "")
    passwords_env = os.environ.get("RAINYUN_PASS", "")
    users = [user.strip() for user in users_env.split('\n') if user.strip()]
    passwords = [pwd.strip() for pwd in passwords_env.split('\n') if pwd.strip()]
    
    if len(users) == len(passwords) and len(users) > 0:
        for user, pwd in zip(users, passwords):
            accounts.append((user, pwd))
    else:
        print("❌ 未找到有效账户配置或数量不匹配")
        exit(1)
    
    results = []
    for i, (user, pwd) in enumerate(accounts, 1):
        print(f"\n{'─'*60}")
        print(f"📋 处理账户 {i}/{len(accounts)}: {user}")
        print(f"{'─'*60}")
        result = sign_in_account(user, pwd, debug=debug, headless=headless)
        results.append(result)
        if result[0]:
            print(f"✅ 账户 {i} 处理完成")
        else:
            print(f"❌ 账户 {i} 处理失败")
    
    success_count = sum(1 for r in results if r[0])
    total_count = len(results)
    
    print(f"\n{'='*60}")
    print(f"📊 签到完成！成功: {success_count}/{total_count}")
    print(f"{'='*60}\n")
    
    if success_count == total_count:
        notification_title = f"✅ 雨云自动签到完成 - 全部成功"
    elif success_count > 0:
        notification_title = f"⚠️ 雨云自动签到完成 - 部分成功 ({success_count}/{total_count})"
    else:
        notification_title = f"❌ 雨云自动签到完成 - 全部失败"
    
    notification_content = f"雨云自动签到结果汇总：\n\n总账户数: {total_count}\n成功账户数: {success_count}\n失败账户数: {total_count - success_count}\n\n详细结果：\n"
    
    for i, (success, user, points, error_msg) in enumerate(results, 1):
        if success:
            notification_content += f"{i}. ✅ {user}\n   积分: {points} | 约 {points / 2000:.2f} 元\n"
        else:
            notification_content += f"{i}. ❌ {user}\n   错误: {error_msg}\n"
    
    if AD_LIST:
        notification_content += "\n" + "=" * 30 + "\n"
        for ad in AD_LIST:
            link = ad.get('link')
            if link and link != 'null':
                notification_content += f"📢 {ad.get('text', '')}\n🔗 {link}\n"
            else:
                notification_content += f"📢 {ad.get('text', '')}\n"
        notification_content += "=" * 30 + "\n"
    elif AD_TEXT:
        notification_content += "\n" + "=" * 30 + "\n"
        if AD_LINK:
            notification_content += f"📢 广告: {AD_TEXT}\n🔗 链接: {AD_LINK}\n"
        else:
            notification_content += f"📢 广告: {AD_TEXT}\n"
        notification_content += "=" * 30 + "\n"
    
    try:
        send(notification_title, notification_content)
        print("✅ 统一通知发送成功")
    except Exception as e:
        print("❌ 发送通知失败")
    
    if AD_LIST:
        print(f"\n{'─'*60}")
        for ad in AD_LIST:
            print(f"📢 {ad.get('text', '')}")
            link = ad.get('link')
            if link and link != 'null':
                print(f"🔗 {link}")
        print(f"{'─'*60}\n")
    elif AD_TEXT:
        print(f"\n{'─'*60}")
        print(f"📢 {AD_TEXT}")
        if AD_LINK:
            print(f"🔗 {AD_LINK}")
        print(f"{'─'*60}\n")
