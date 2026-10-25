import asyncio
from playwright.async_api import async_playwright
import time

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel='msedge', headless=True)
        page = await browser.new_page()

        username = f"dr_rajesh_{int(time.time())}"
        await page.goto("http://127.0.0.1:5000/register")
        await page.fill("input#username", username)
        await page.fill("input#full_name", "Dr Rajesh")
        await page.fill("input#email", f"{username}@hospital.org")
        await page.fill("input#password", "Password123!")
        await page.click('button[type="submit"]')
        await page.wait_for_timeout(400)

        await page.goto("http://127.0.0.1:5000/login")
        await page.fill("input#username", username)
        await page.fill("input#password", "Password123!")
        await page.click('button[type="submit"]')
        await page.wait_for_timeout(400)

        await page.goto("http://127.0.0.1:5000/upload_fingerprint")
        await page.wait_for_timeout(800)

        # Upload a real BMP fingerprint image from Test Images
        file_input = await page.query_selector('input#fingerprintImageInput')
        await file_input.set_input_files('C:/Users/miche/Downloads/Code (2)/Code/Code/Test Images/cluster_0_26.BMP')
        await page.wait_for_timeout(400)

        print("Clicking Extract Biometrics Button...")
        await page.click('button#extractBtn')
        await page.wait_for_selector('#extractedResultCard:not(.hidden)', timeout=15000)
        await page.wait_for_timeout(1000)

        screenshot_path = "C:/Users/miche/AppData/Local/hermes/cache/screenshots/08_mnc_extraction_with_5_stages.png"
        await page.screenshot(path=screenshot_path, full_page=True)
        print(f"MEDIA:{screenshot_path}")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())
