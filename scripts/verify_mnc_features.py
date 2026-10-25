import asyncio
from playwright.async_api import async_playwright
import os
import time

async def test_full_mnc_flow():
    screenshot_dir = 'C:/Users/miche/AppData/Local/hermes/cache/screenshots'
    os.makedirs(screenshot_dir, exist_ok=True)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel='msedge', headless=True)
        page = await browser.new_page(viewport={'width': 1280, 'height': 850})
        
        # 1. Register & Login
        uname = f"dr_rajesh_{int(time.time())}"
        print(f"1. Registering as {uname}...")
        await page.goto('http://127.0.0.1:5000/register')
        await page.fill('input#username', uname)
        await page.fill('input#email', f"{uname}@hospital.org")
        await page.fill('input#password', 'password123')
        await page.fill('input#full_name', 'Dr. Rajesh')
        await page.click('button[type="submit"]')
        await page.wait_for_timeout(1000)
        
        # 2. Login
        print("2. Logging in...")
        await page.goto('http://127.0.0.1:5000/login')
        await page.fill('input#username', uname)
        await page.fill('input#password', 'password123')
        await page.click('button[type="submit"]')
        await page.wait_for_timeout(1000)
        print(f"Logged in, URL is: {page.url}")
        
        # 3. Go to /upload_fingerprint
        print("3. Checking Live Sensor...")
        await page.goto('http://127.0.0.1:5000/upload_fingerprint')
        await page.wait_for_timeout(1500)
        p1 = f"{screenshot_dir}/01_mnc_sensor_detected.png"
        await page.screenshot(path=p1)
        print(f"MEDIA:{p1}")
        
        # 4. Trigger Live Hardware Scan
        print("4. Triggering Live Biometric Hardware Scan...")
        await page.click('button#liveScanBtn')
        await page.wait_for_timeout(2500)
        p2 = f"{screenshot_dir}/02_mnc_live_scanned_stages.png"
        await page.screenshot(path=p2)
        print(f"MEDIA:{p2}")
        
        # 5. Proceed to Step 2
        print("5. Proceeding to Step 2 (Vitals Form)...")
        await page.goto('http://127.0.0.1:5000/prediction_form')
        await page.wait_for_timeout(1000)
        await page.fill('input#age', '48')
        await page.fill('input#bmi', '29.2')
        await page.select_option('select#family_history', 'yes')
        await page.select_option('select#model_choice', 'Auto-Select (AI Metric-Driven)')
        p3 = f"{screenshot_dir}/03_mnc_auto_optimizer_form.png"
        await page.screenshot(path=p3)
        print(f"MEDIA:{p3}")
        
        # 6. Submit Prediction
        print("6. Submitting Prediction with Dynamic ML Selection...")
        await page.click('button[type="submit"]')
        await page.wait_for_timeout(2000)
        print(f"Prediction Complete, URL is: {page.url}")
        p4 = f"{screenshot_dir}/04_mnc_dossier_with_audit.png"
        await page.screenshot(path=p4)
        print(f"MEDIA:{p4}")
        
        await browser.close()
        print("=== MNC-GRADE FULL PIPELINE COMPLETED SUCCESSFULLY ===")

asyncio.run(test_full_mnc_flow())
