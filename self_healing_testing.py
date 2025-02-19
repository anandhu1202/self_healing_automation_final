import os
import json
import time
import pickle
import numpy as np
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from xgboost import XGBClassifier

# --- Global Settings & Persistence Files ---
GLOBAL_GOLDEN_FILE = "global_golden.json"
TRAINING_DATA_FILE = "training_data.pkl"
MODEL_FILE = "ml_model.pkl"
MIN_SAMPLES_FOR_MODEL = 5  # Minimum examples needed to (re)train the model

# --- Persistence Functions for Training Data and Model ---
def save_training_data(training_data, training_labels):
    with open(TRAINING_DATA_FILE, "wb") as f:
        pickle.dump((training_data, training_labels), f)
    print("Training data saved.")

def load_training_data():
    if os.path.exists(TRAINING_DATA_FILE):
        with open(TRAINING_DATA_FILE, "rb") as f:
            training_data, training_labels = pickle.load(f)
        print("Training data loaded.")
        return training_data, training_labels
    else:
        print("No training data found. Starting fresh.")
        return [], []

def save_model(model):
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    print("ML model saved.")

def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        print("ML model loaded.")
        return model
    else:
        print("No ML model found. Will train a new one when enough data is collected.")
        return None

def load_global_golden_data():
    if os.path.exists(GLOBAL_GOLDEN_FILE):
        with open(GLOBAL_GOLDEN_FILE, "r") as f:
            return json.load(f)
    return {}

def store_global_golden_data(data):
    with open(GLOBAL_GOLDEN_FILE, "w") as f:
        json.dump(data, f, indent=4)

# --- Golden Identifier ---
def generate_golden_identifier(driver, original_xpath, element=None):
    """
    Generates a golden identifier based on the page context and stable attributes.
    """
    page_key = driver.title.replace(" ", "_")
    
    if element:
        # Prefer data-testid if available
        candidate = element.get_attribute("data-testid")
        if candidate:
            return f"{page_key}_golden_{candidate}"
        # Else, try id or name
        candidate = element.get_attribute("id") or element.get_attribute("name")
        if candidate:
            return f"{page_key}_golden_{candidate}"
        # Else, use tag and class
        candidate_classes = element.get_attribute("class")
        if candidate_classes:
            classes = candidate_classes.strip().replace(" ", "_")
            return f"{page_key}_golden_{element.tag_name}_{classes}"
        # Or use a snippet of text if short
        candidate_text = element.text.strip()
        if candidate_text and len(candidate_text) < 20:
            return f"{page_key}_golden_{element.tag_name}_{candidate_text.replace(' ', '_')}"
    
    sanitized_xpath = original_xpath.replace("/", "_").replace("[", "").replace("]", "").replace("@", "")
    return f"{page_key}_golden_{sanitized_xpath}"

# --- Core Self-Healing Functions ---
def capture_element_attributes(element):
    """
    Capture stable attributes of the element and its parent.
    """
    attributes = {
        'tag': element.tag_name,
        'id': element.get_attribute('id'),
        'class': element.get_attribute('class'),
        'name': element.get_attribute('name'),
        'data-testid': element.get_attribute('data-testid'),
        'text': element.text.strip()
    }
    if element.tag_name.lower() == 'div':
        attributes['innerHTML'] = element.get_attribute('innerHTML').strip()
    try:
        parent = element.find_element(By.XPATH, "..")
        attributes['parent'] = {
            'tag': parent.tag_name,
            'id': parent.get_attribute('id'),
            'class': parent.get_attribute('class'),
            'name': parent.get_attribute('name'),
            'data-testid': parent.get_attribute('data-testid')
        }
    except Exception:
        attributes['parent'] = None
    return attributes

def compute_similarity(golden, candidate):
    score = 0
    if candidate.tag_name.lower() != golden['tag'].lower():
        return 0
    if candidate.get_attribute("id") == golden.get("id"):
        score += 10
    if candidate.get_attribute("name") == golden.get("name"):
        score += 10
    if candidate.get_attribute("data-testid") == golden.get("data-testid"):
        score += 8
    if candidate.get_attribute("class") == golden.get("class"):
        score += 5
    candidate_text = candidate.text.strip()
    if candidate_text and golden.get("text") and golden["text"] in candidate_text:
        score += 3
    if golden.get("innerHTML"):
        candidate_innerHTML = candidate.get_attribute("innerHTML").strip()
        if candidate_innerHTML and golden["innerHTML"] in candidate_innerHTML:
            score += 2
    try:
        candidate_parent = candidate.find_element(By.XPATH, "..")
        golden_parent = golden.get("parent")
        if golden_parent:
            if candidate_parent.tag_name.lower() == golden_parent.get("tag", "").lower():
                score += 2
            if candidate_parent.get_attribute("id") == golden_parent.get("id"):
                score += 5
            if candidate_parent.get_attribute("class") == golden_parent.get("class"):
                score += 3
    except Exception:
        pass
    return score

def extract_features(golden, candidate):
    """
    Extract features as binary flags and normalized scores.
    Features:
      f0: id match
      f1: name match
      f2: data-testid match
      f3: class match
      f4: text containment
      f5: innerHTML containment
      f6: normalized heuristic similarity (raw_score / 48)
      f7: parent's tag match
      f8: parent's id match
      f9: parent's class match
    """
    f0 = 1 if candidate.get_attribute("id") == golden.get("id") else 0
    f1 = 1 if candidate.get_attribute("name") == golden.get("name") else 0
    f2 = 1 if candidate.get_attribute("data-testid") == golden.get("data-testid") else 0
    f3 = 1 if candidate.get_attribute("class") == golden.get("class") else 0
    candidate_text = candidate.text.strip()
    f4 = 1 if candidate_text and golden.get("text") and golden["text"] in candidate_text else 0
    f5 = 0
    if golden.get("innerHTML"):
        candidate_innerHTML = candidate.get_attribute("innerHTML")
        if candidate_innerHTML and golden["innerHTML"] in candidate_innerHTML:
            f5 = 1
    raw_score = compute_similarity(golden, candidate)
    f6 = raw_score / 48.0  # Normalize (max score estimated at 48)
    f7, f8, f9 = 0, 0, 0
    try:
        candidate_parent = candidate.find_element(By.XPATH, "..")
        golden_parent = golden.get("parent")
        if golden_parent:
            f7 = 1 if candidate_parent.tag_name.lower() == golden_parent.get("tag", "").lower() else 0
            f8 = 1 if candidate_parent.get_attribute("id") == golden_parent.get("id") else 0
            f9 = 1 if candidate_parent.get_attribute("class") == golden_parent.get("class") else 0
    except Exception:
        pass
    return [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9]

def get_all_candidates(driver, golden):
    tag = golden.get("tag", "*")
    return driver.find_elements(By.TAG_NAME, tag)

def generate_selector(candidate):
    candidate_id = candidate.get_attribute("id")
    if candidate_id:
        return f"//*[@id='{candidate_id}']"
    candidate_testid = candidate.get_attribute("data-testid")
    if candidate_testid:
        return f"//*[@data-testid='{candidate_testid}']"
    candidate_placeholder = candidate.get_attribute("placeholder")
    if candidate_placeholder:
        return f"//{candidate.tag_name}[@placeholder='{candidate_placeholder}']"
    candidate_text = candidate.text.strip()
    if candidate_text:
        return f"//{candidate.tag_name}[contains(text(), '{candidate_text}')]"
    candidate_class = candidate.get_attribute("class")
    if candidate_class:
        classes = candidate_class.split()
        return candidate.tag_name + "." + ".".join(classes)
    return f"//{candidate.tag_name}"

def self_heal_selector(driver, golden, training_data, training_labels, model):
    candidates = get_all_candidates(driver, golden)
    if not candidates:
        raise Exception("No candidate elements found with tag: " + golden.get("tag", "*"))
    
    features_list = []
    candidate_list = []
    heuristic_scores = []
    for candidate in candidates:
        features = extract_features(golden, candidate)
        features_list.append(features)
        candidate_list.append(candidate)
        heuristic_scores.append(compute_similarity(golden, candidate))
    
    best_index_heuristic = int(np.argmax(heuristic_scores))
    # Create labels: mark the best heuristic candidate as 1, rest 0
    labels = [1 if i == best_index_heuristic else 0 for i in range(len(candidates))]
    
    # Append current examples to our training data
    training_data.extend(features_list)
    training_labels.extend(labels)
    
    # If enough data, train/update the model
    if len(training_labels) >= MIN_SAMPLES_FOR_MODEL:
        clf = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
        clf.fit(np.array(training_data), np.array(training_labels))
        model = clf
        save_training_data(training_data, training_labels)
        save_model(model)
    else:
        print("Not enough training data for ML; using heuristic ranking.")
    
    # Use ML model if available; otherwise, fallback to heuristic
    if model is not None:
        probabilities = model.predict_proba(np.array(features_list))[:, 1]
        best_index_model = int(np.argmax(probabilities))
        chosen_candidate = candidate_list[best_index_model]
        print("XGBoost selected candidate with probability:", probabilities[best_index_model])
    else:
        chosen_candidate = candidate_list[best_index_heuristic]
        print("Using heuristic best candidate.")
    
    new_selector = generate_selector(chosen_candidate)
    try:
        driver.find_element(By.XPATH, new_selector)
        print("Self-healing successful. New selector:", new_selector)
        return new_selector, chosen_candidate, training_data, training_labels, model
    except NoSuchElementException:
        raise Exception("Self-healing failed: the generated selector did not locate an element.")

def get_updated_locator(driver, original_xpath, golden, training_data, training_labels, model):
    try:
        driver.find_element(By.XPATH, original_xpath)
        # Return values consistently
        return original_xpath, None, training_data, training_labels, model
    except NoSuchElementException:
        print("Original locator failed; triggering self-healing.")
        return self_heal_selector(driver, golden, training_data, training_labels, model)

# --- Self-Healing Agent Class ---
class SelfHealingAgent:
    def __init__(self, driver):
        self.driver = driver
        self.global_data = load_global_golden_data()
        self.page_key = self.driver.title.replace(" ", "_")
        if self.page_key not in self.global_data:
            self.global_data[self.page_key] = {}
            store_global_golden_data(self.global_data)
        self.training_data, self.training_labels = load_training_data()
        self.model = load_model()
    
    def capture_golden_if_missing(self, original_xpath):
        golden_id = generate_golden_identifier(self.driver, original_xpath)
        if golden_id not in self.global_data[self.page_key]:
            try:
                element = self.driver.find_element(By.XPATH, original_xpath)
                golden = capture_element_attributes(element)
                self.global_data[self.page_key][golden_id] = golden
                store_global_golden_data(self.global_data)
                print(f"Captured golden reference for {golden_id}: {golden}")
            except Exception as e:
                print(f"Failed to capture golden reference for {golden_id}: {e}")
        return golden_id
    
    def locate_element(self, original_xpath):
        golden_identifier = self.capture_golden_if_missing(original_xpath)
        golden = self.global_data[self.page_key].get(golden_identifier)
        try:
            element = self.driver.find_element(By.XPATH, original_xpath)
            print("Element found using original locator.")
            return element
        except NoSuchElementException:
            result = get_updated_locator(self.driver, original_xpath, golden, self.training_data, self.training_labels, self.model)
            new_locator, _, self.training_data, self.training_labels, self.model = result
            print(f"Agent updated locator: {new_locator}")
            return self.driver.find_element(By.XPATH, new_locator)
    
    def click_element(self, original_xpath):
        element = self.locate_element(original_xpath)
        element.click()
    
    def fill_field(self, original_xpath, value):
        element = self.locate_element(original_xpath)
        element.clear()
        element.send_keys(value)
