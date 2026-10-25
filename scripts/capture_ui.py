import asyncio
from playwright.async_api import async_playwright
import os
import time

async def capture_all():
    screenshot_dir = 'C:/Users/miche/AppData/Local/hermes/cache/screenshots'
    os.makedirs(screenshot_dir, exist_ok=True)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel='msedge', headless=True)
        context = await browser.new_context(viewport={'width': 1280, 'height': 850})
        page = await context.new_page()
        
        # 1. Home Page
        await page.goto('http://127.0.0.1:5000/')
        await page.wait_for_timeout(1000)
        p1 = f"{screenshot_dir}/01_pro_home.png"
        await page.screenshot(path=p1)
        print(f"MEDIA:{p1}")
        
        # 2. Benchmark Page
        await page.goto('http://127.0.0.1:5000/model_benchmark')
        await page.wait_for_timeout(1000)
        p2 = f"{screenshot_dir}/02_pro_benchmark.png"
        await page.screenshot(path=p2)
        print(f"MEDIA:{p2}")
        
        # 3. Register & Login
        uname = f"dr_raj_{int(time.time())}"
        await page.goto('http://127.0.0.1:5000/register')
        await page.fill('input#username', uname)
        await page.fill('input#email', f"{uname}@hospital.org")
        await page.fill('input#password', 'secure_pass123')
        await page.fill('input#full_name', 'Dr. Rajesh')
        await page.click('button[type="submit"]')
        await page.wait_for_timeout(1000)
        
        # Login
        await page.fill('input#username', uname)
        await page.fill('input#password', 'secure_pass123')
        await page.click('button[type="submit"]')
        await page.wait_for_timeout(1000)
        
        # 4. Upload Fingerprint Page
        test_img = 'C:/Users/miche/Downloads/Code (2)/Code/Code/Test Images/cluster_0_26.BMP'
        await page.set_input_files('#fingerprintImageInput', test_img)
        await page.wait_for_timeout(800)
        await page.click('#extractBtn')
        await page.wait_for_selector('#resultCard:not(.hidden)', timeout=15000)
        p3 = f"{screenshot_dir}/03_pro_extracted_stages.png"
        await page.screenshot(path=p3)
        print(f"MEDIA:{p3}")
        
        # 5. Prediction Form
        await page.click('a:has-text("Proceed to Step 2")')
        await page.wait_for_selector('form[action="/predict_diabetes"]')
        await page.fill('input#age', '46')
        await page.fill('input#bmi', '28.8')
        await page.select_option('select#family_history', 'yes')
        p4 = f"{screenshot_dir}/04_pro_prediction_form.png"
        await page.screenshot(path=p4)
        print(f"MEDIA:{p4}")
        
        # 6. Execute Prediction & View Result Dossier
        await page.click('button[type="submit"]')
        await page.wait_for_url('**/result*', timeout=15000)
        await page.wait_for_timeout(1500)
        p5 = f"{screenshot_dir}/05_pro_result_dossier.png"
        await page.screenshot(path=p5)
        print(f"MEDIA:{p5}")
        
        # 7. Dashboard Analytics
        await page.goto('http://127.0.0.1:5000/dashboard')
        await page.wait_for_timeout(1200)
        p6 = f"{screenshot_dir}/06_pro_dashboard.png"
        await page.screenshot(path=p6)
        print(f"MEDIA:{p6}")
        
        await browser.close()
        print("ALL UI SCREENSHOTS CAPTURED SUCCESSFULLY!")

if __name__ == '__main__':
    asyncio.run(capture_all())
