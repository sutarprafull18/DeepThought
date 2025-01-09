import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL to scrape
url = "https://www.nestle.in/about-us"

# Function to scrape data
def scrape_data(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract company name (assuming it's in the title tag)
    company_name = soup.title.string if soup.title else "N/A"


    company_type = []
    product_categories = []
    manufacturer =[]
    brand = []
    distributor = []
    f_and_b = []

    brand_element = soup.find('h2', id="block-da-vinci-code-downloads-menu")
    if brand_element:
        brand = brand_element.get_text(strip=True)

    company_type_element = soup.find('div', class_='company-type-class')
    if company_type_element:
        company_type = company_type_element.get_text(strip=True)

    manufacturer_element = soup.find('p')
    if manufacturer_element:
        manufacturer = manufacturer_element.get_text(strip=True)

    distributor_element = soup.find('h2', id="block-da-vinci-code-media-menu")  # Example class name
    if distributor_element:
        distributor = distributor_element.get_text(strip=True)

    product_categories_element = soup.find('div', class_='product-categories-class')  # Example class name
    if product_categories_element:
        product_categories = product_categories_element.get_text(strip=True)



    return {
        'Company Name': company_name,
        'Company Type': company_type,
        'Product Categories': product_categories,
        'Manufacturer': manufacturer,
        'Brand': brand,
        'Distributor': distributor,

    }

# Scrape data from the URL
data = scrape_data(url)
if data:
    df = pd.DataFrame([data])
    df.to_excel('nestle_data.xlsx', index=False)
    print("Data saved to nestle_data.xlsx")
else:
    print("Failed to scrape data")
