import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from bs4 import BeautifulSoup
import time
import requests
import json
import random
import os

# Predefined User-Agent headers for different browsers
headers_list = [
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"},
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:58.0) Gecko/20100101 Firefox/58.0"},
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36 Edge/92.0.902.73"},
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1; rv:93.0) Gecko/20100101 Firefox/93.0"},
    {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"},
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1; rv:60.0) Gecko/20100101 Firefox/60.0"},
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36"},
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36"},
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.5195.52 Safari/537.36"},
    {
        "User-Agent": "Mozilla/5.0 (Linux; Android 10; Pixel 3 XL) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Mobile Safari/537.36"},
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36"},
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/100.0.1185.50 Safari/537.36"},
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1; rv:90.0) Gecko/20100101 Firefox/90.0"},
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0"},
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:80.0) Gecko/20100101 Firefox/80.0"},
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.3; Trident/7.0; AS; rv:11.0) like Gecko"},
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36"},
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36 Edge/16.16299"},
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"},
    {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"},
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.54 Safari/537.36"},
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36"},
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36"},
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36 Edge/94.0.992.50"},
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/82.0.4085.0 Safari/537.36"},
    {
        "User-Agent": "Mozilla/5.0 (Linux; Android 9; Pixel 3 XL) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.119 Mobile Safari/537.36"},
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Trident/7.0; AS; rv:11.0) like Gecko"},
    {
        "User-Agent": "Mozilla/5.0 (Linux; Android 11; OnePlus 8 Pro) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36"},
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.3; Trident/7.0; AS; rv:11.0) like Gecko"},
    {
        "User-Agent": "Mozilla/5.0 (Linux; U; Android 10; en-US; Pixel 4 XL Build/QP1A.190711.020) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.101 Mobile Safari/537.36"}
]

# Function to scrape article links from a given ticker
def get_article_links(ticker, ticker_name):
    # Configure Chrome browser options
    chrome_options = Options()

    # Set User-Agent to simulate device login
    mobile_user_agent = random.choice(headers_list)["User-Agent"]
    chrome_options.add_argument(f"user-agent={mobile_user_agent}")

    # Headless mode, running in the background
    chrome_options.add_argument("headless")

    # Configure and launch the WebDriver
    driver = webdriver.Chrome(options=chrome_options)

    # Website
    url = f"https://stcn.com/article/search.html?search_type=news&keyword={ticker_name}&uncertainty=1&sorter=time"
    driver.get(url)

    # Wait for initial page load to prevent failure due to slow network speed
    time.sleep(3)

    count = 0
    retries = 0
    # Initialize the page height
    last_height = driver.execute_script("return document.body.scrollHeight")
    # Scroll until no new content is loaded
    while True:
        try:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            time.sleep(2)

            new_height = driver.execute_script("return document.body.scrollHeight")

            # Check if the page height has changed
            if new_height == last_height:
                retries += 1
                print(f"Page height did not change, retry count：{retries}")
                if retries >= 3:
                    print("Retried 3 times, page height still unchanged, stopping scroll")
                    break
                time.sleep(2)
            else:
                count += 1
                print(f"Page refresh count：{count}")
                if (count == 200):
                    break
                if (count % 15 == 0):
                    time.sleep(5)
                retries = 0
                # Update page height
                last_height = new_height

        except Exception as e:
            print(f"error：{e}")
            break

    html_code = driver.page_source

    # Parse dynamically loaded HTML
    soup = BeautifulSoup(html_code, 'html.parser')

    # Get all <div class="tt"> tags
    divs = soup.find_all('div', class_='tt')

    base_url = "https://www.stcn.com/"

    article_links = []

    # Iterate through each <a> tag inside the <div class="tt"> tags
    for div in divs:
        a_tag = div.find('a', href=True)
        if a_tag:
            href = a_tag['href']
            if "article" in href:
                full_url = base_url + href
                article_links.append(full_url)

    print(f"{ticker_name}News link count：{len(article_links)}")

    output_file = f"../news_link/{ticker}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(article_links, f, ensure_ascii=False, indent=4)

    # close website
    driver.quit()

# Function to fetch news from a given ticker
def fetch_news(ticker):
    all_news = []

    with open(f"../news_link/{ticker}.json", 'r', encoding='utf-8') as file:
        article_links = json.load(file)
    article_links_len = len(article_links)
    count = 0
    consecutive_failures = 0
    fetch_failed = False

    for article_link in article_links:
        try:

            chrome_options = Options()
            chrome_options.add_argument("headless")
            # use random User-Agent
            chrome_options.add_argument(f"user-agent={random.choice(headers_list)['User-Agent']}")

            driver = webdriver.Chrome(options=chrome_options)

            count += 1
            if count % 120 == 0:
                # Pause for 10 seconds every 120 requests to prevent being blocked due to high frequency
                time.sleep(10)

            print(f"scraping{count}/{article_links_len}: {article_link}")

            try:
                driver.get(article_link)
                time.sleep(1.2)


                soup = BeautifulSoup(driver.page_source, 'html.parser')

                detail_content = soup.find('div', class_='detail-content')

                # Get plain text content (excluding hyperlinks)
                for a_tag in detail_content.find_all('a'):
                    # Replace the entire <a> tag with its text content (remove hyperlinks)
                    a_tag.replace_with(a_tag.text)

                detail_content = detail_content.text.strip()

                detail_title = soup.find('div', class_='detail-title').text.strip()

                detail_info = soup.find('div', class_='detail-info')

                # Get the last <span> tag
                time_span = detail_info.find_all('span')[-1]

                # get timestamp
                time_text = time_span.get_text(strip=True)

                # save data
                all_news.append({
                    "time": time_text,
                    "title": detail_title,
                    "content": detail_content,
                    "link": article_link
                })

                driver.quit()

                consecutive_failures = 0

                fetch_failed = False
                # print(all_news[-1])
            except Exception as e:
                print(f"scraping failed：{e}")
                driver.quit()
                consecutive_failures += 1
                fetch_failed = True
                if consecutive_failures >= 10:
                    print("Failed to open 10 consecutive links, stopping scraping")
                    break  # 终止循环
        except Exception as e:
            print(f"Failed to start WebDriver: {e}")
            consecutive_failures += 1
            fetch_failed = True
            if consecutive_failures >= 10:
                print("Failed to open 10 consecutive links, stopping scraping")
                break

    # saved to json
    if fetch_failed:
        output_file = f"../news/{ticker}_failed.json"
    else:
        output_file = f"../news/{ticker}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_news, f, ensure_ascii=False, indent=4)

    print(f"Completed scraping {len(all_news)} news articles, saved to {output_file}")


def main():
    data = pd.read_csv('Corrected_CSV_File_Last_Date_Summary.csv', encoding='utf-8')
    tickers = data['code'].tolist()
    ticker_names = data['code_name'].tolist()
    for ticker, ticker_name in zip(tickers, ticker_names):
        if not os.path.exists(f"../news/{ticker}.json"):
            print(f"{ticker}: {ticker_name}")

            try:
                # get news link
                get_article_links(ticker, ticker_name)
            except Exception as e:
                print(f"Failed to get article links for {ticker}: {e}")

                with open(f"../news/{ticker}_failed.json", 'w') as f:
                    json.dump({"error": "Failed to get article links"}, f)
                continue

            # news scraper
            fetch_news(ticker)

if __name__ == '__main__':
    main()








