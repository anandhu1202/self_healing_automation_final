# test_cases_agent_extended.py

import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from self_healing_agent import SelfHealingAgent

def test_page1(driver, agent):
    # Page 1: Form page
    form_data = {
        "name": "John Doe",
        "email": "john.doe@example.com",
        "message": "This is a test message."
    }
    
    for field, value in form_data.items():
        tag = "textarea" if field.lower() == "message" else "input"
        original_xpath = f"//{tag}[@id='{field}']"
        agent.fill_field(original_xpath, value)
        time.sleep(0.5)
    
    # Submit the form
    submit_xpath = "//form[@id='userDataForm']//input[@type='submit']"
    agent.click_element(submit_xpath)
    time.sleep(2)
    
    # Click additional buttons on Page 1
    actions = [
        {"name": "Submit", "xpath": "//*[@id='submit-btn']"},
        {"name": "Cancel", "xpath": "//*[@id='cancel-btn']"},
        {"name": "Next", "xpath": "//*[@id='next-btn']"}
    ]
    for action in actions:
        agent.click_element(action["xpath"])
        time.sleep(1)

def test_page3(driver, agent):
    # Page 3: Login Page Example
    # Assume the login page has an email field, a password field, and a login button.
    login_data = {
        "email": "user@example.com",
        "password": "password123"
    }
    # Fill the email field (assume id="login-email")
    agent.fill_field("//input[@id='login-email']", login_data["email"])
    time.sleep(0.5)
    # Fill the password field (assume id="login-password")
    agent.fill_field("//input[@id='login-password']", login_data["password"])
    time.sleep(0.5)
    # Click the login button (assume id="login-btn")
    agent.click_element("//button[@id='login-btn']")
    time.sleep(2)

def test_page4(driver, agent):
    # Page 4: Dashboard Navigation Example
    # Assume the dashboard has navigation links for Profile, Settings, and Logout.
    actions = [
        {"name": "Profile", "xpath": "//a[@id='nav-profile']"},
        {"name": "Settings", "xpath": "//a[@id='nav-settings']"},
        {"name": "Logout", "xpath": "//a[@id='nav-logout']"}
    ]
    for action in actions:
        agent.click_element(action["xpath"])
        time.sleep(1)

def main():
    driver = webdriver.Chrome()
    driver.maximize_window()
    
    try:
        # --- Page 1 Test ---
        driver.get("http://localhost:8000/page1.html")
        time.sleep(2)
        agent = SelfHealingAgent(driver)
        test_page1(driver, agent)
        time.sleep(2)
        
        # --- Page 4 Test (Dashboard) ---
        driver.get("http://localhost:8000/page4.html")
        time.sleep(2)
        test_page4(driver, agent)
        time.sleep(2)
        
        # Handle any alert from Page 4 before moving on
        try:
            alert = driver.switch_to.alert
            print("Alert found before navigating to Page 3: " + alert.text)
            alert.accept()
            print("Alert accepted.")
        except Exception as e:
            print("No alert present, proceeding to Page 3.")
        
        # --- Page 3 Test (Login Page) ---
        driver.get("http://localhost:8000/page3.html")
        time.sleep(2)
        test_page3(driver, agent)
        time.sleep(2)
        
    finally:
        driver.quit()

if __name__ == "__main__":
    main()