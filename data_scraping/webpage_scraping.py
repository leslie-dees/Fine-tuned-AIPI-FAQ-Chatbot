import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin
import os

def scrape_website_for_plaintext(url):
    # Provided url scrapes the website for plaintext content, does not fetch hyperlinks
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        text_content = ' '.join([element.get_text().strip() for element in soup.find_all(string=True)])
        # Replace large chunks of whitespace with single newline characters
        text_content = re.sub(r'\s{2,}', '\n\n', text_content)
        return text_content
    else:
        print(f"Failed to retrieve the webpage {url}")
        return None

def scrape_website_for_links(url):
    # Finds all hyperlinks and extracts their URLs
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        links = [a['href'] for a in soup.find_all('a', href=True)]
        unique_links = set()
        base_url = response.url
        for link in links:
            if '#' not in link:
                if link.startswith('https://') or link.startswith('http://'):
                    if not any(keyword in link for keyword in ['youtube', 'twitter', 'facebook', 'instagram']):
                        unique_links.add(link)
                elif link.startswith('/'):
                    absolute_link = urljoin(base_url, link)
                    if not any(keyword in absolute_link for keyword in ['youtube', 'twitter', 'facebook', 'instagram', 'linkedin']):
                        unique_links.add(absolute_link)
        return list(unique_links)
    else:
        print(f"Failed to retrieve the webpage for {url}")
        return None


def sanitize_filename(filename):
    # Remove characters not allowed in filenames
    sanitized_filename = re.sub(r'[\\/:*?"<>|]', '_', filename)
    return sanitized_filename

def save_website_plaintext(url, plaintext, folder_path='raw_website_data'):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Sanitize the URL to create a valid filename
    sanitized_filename = sanitize_filename(url)
    
    filename = os.path.join(folder_path, sanitized_filename + '.txt')
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(plaintext)
    print(f"Plaintext content for {url} saved successfully.")

def scrape_website_and_save_plaintext(url):
    # Scrape plaintext for the main URL
    plaintext = scrape_website_for_plaintext(url)
    if plaintext:
        save_website_plaintext(url, plaintext)

    # Scrape plaintext for all hyperlinks found on the main URL
    links = scrape_website_for_links(url)
    if links:
        for link in links:
            text_content = scrape_website_for_plaintext(link)
            if text_content:
                
                save_website_plaintext(link, text_content)

def consolidate_website_data(folder_path='raw_website_data', output_file='consolidated_aipi_website_data.txt'):
    with open(folder_path+'/'+output_file, 'w', encoding='utf-8') as consolidated_file:
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    file_content = file.read()
                    consolidated_file.write(file_content)
                    consolidated_file.write('\n\n')


if __name__ == "__main__":
    # URLs to scrape
    urls = [
        "https://ai.meng.duke.edu/",
        "https://ai.meng.duke.edu/degree"
    ]

    # Scrape each website and save plaintext content
    for url in urls:
        scrape_website_and_save_plaintext(url)

    # Consolidate the website data into a single file
    consolidate_website_data()