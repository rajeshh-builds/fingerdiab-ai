import asyncio
from playwright.async_api import async_playwright
import os
import time

async def main():
    screenshot_dir = 'C:/Users/miche/AppData/Local/hermes/cache/screenshots'
    os.makedirs(screenshot_dir, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(channel='msedge', headless=True)
        context = await browser.new_context(viewport={'width': 1440, 'height': 900})
        page = await context.new_page()

        print("1. Visiting Homepage...")
        await page.goto('http://127.0.0.1:5005/', wait_until='networkidle')
        await page.screenshot(path=f"{screenshot_dir}/10_classic_homepage.png", full_page=True)
        print(f"MEDIA:{screenshot_dir}/10_classic_homepage.png")

        # Create unique user
        uid = int(time.time())
        username = f"dr_rajesh_{uid}"
        password = f"SecurePass{uid}!"

        print(f"2. Registering as {username}...")
        await page.goto('http://127.0.0.1:5005/register', wait_until='networkidle')
        await page.fill('#full_name', 'Dr. Rajesh Kumar')
        await page.fill('#username', username)
        await page.fill('#email', f"{username}@healthcare.org")
        await page.fill('#password', password)
        await page.screenshot(path=f"{screenshot_dir}/11_classic_register.png")
        print(f"MEDIA:{screenshot_dir}/11_classic_register.png")
        await page.click('button[type="submit"]')
        await page.wait_for_timeout(1000)

        print("3. Logging in...")
        if 'login' in page.url:
            await page.fill('#username', username)
            await page.fill('#password', password)
            await page.screenshot(path=f"{screenshot_dir}/12_classic_login.png")
            print(f"MEDIA:{screenshot_dir}/12_classic_login.png")
            await page.click('button[type="submit"]')
            await page.wait_for_timeout(1000)

        print("4. Step 1: Upload Fingerprint...")
        await page.goto('http://127.0.0.1:5005/upload_fingerprint', wait_until='networkidle')
        await page.screenshot(path=f"{screenshot_dir}/13_classic_step1_initial.png")
        print(f"MEDIA:{screenshot_dir}/13_classic_step1_initial.png")

        print("Loading Whorl Archetype...")
        await page.click('button:has-text("Whorl Pattern")')
        await page.wait_for_timeout(600)

        print("Extracting Biometrics...")
        await page.click('button#extractBtn')
        await page.wait_for_selector('#extractedResultCard:not(.hidden)', timeout=15000)
        await page.wait_for_timeout(1000)
        await page.screenshot(path=f"{screenshot_dir}/14_classic_step1_extracted_stages.png", full_page=True)
        print(f"MEDIA:{screenshot_dir}/14_classic_step1_extracted_stages.png")

        print("Proceeding to Step 2...")
        await page.click('a#proceedStep2Btn')
        await page.wait_for_url('**/prediction_form**', timeout=15000)
        await page.wait_for_timeout(1000)

        print("5. Step 2: Clinical Vitals...")
        print(f"Current URL on Step 2: {page.url}")
        await page.fill('#age', '52')
        await page.fill('#bmi', '29.8')
        await page.select_option('#family_history', 'yes')
        await page.screenshot(path=f"{screenshot_dir}/15_classic_step2_clinical_vitals.png", full_page=True)
        print(f"MEDIA:{screenshot_dir}/15_classic_step2_clinical_vitals.png")

        print("Submitting Analysis...")
        await page.click('button[type="submit"]')
        await page.wait_for_url('**/result**', timeout=15000)
        await page.wait_for_timeout(1000)

        print("6. Step 3: Diagnostic Dossier...")
        await page.screenshot(path=f"{screenshot_dir}/16_classic_step3_dossier.png", full_page=True)
        print(f"MEDIA:{screenshot_dir}/16_classic_step3_dossier.png")

        await browser.close()
        print("=== ALL CLASSIC MNC VERIFICATION SCREENSHOTS CAPTURED SUCCESSFULLY ===")

if __name__ == '__main__':
    asyncio.run(main())