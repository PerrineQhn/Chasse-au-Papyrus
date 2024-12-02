import requests
from bs4 import BeautifulSoup
import csv

def get_papyrus_ids(csv_corpus: csv):
    """ 
    Get the papyrus ids from a CSV file
    """
    ids = []
    with open(csv_corpus, 'r') as file:
        next(file)  # Skip the header
        for line in file:
            # Parse the CSV content of the page
            papyrus_infos = line.split(',')
            papyrus_ids = papyrus_infos[0]
            if papyrus_ids:  # Verify that the papyrus_ids is not empty
                parts = papyrus_ids.split()
                if len(parts) == 2:  # Verify that there are two parts
                    _, num = parts  # Skip the first part (TM)
                    ids.append(num)
    return ids

def generate_papyrus_urls(papyrus_ids: list):
    base_url = "https://www.trismegistos.org/text/"
    urls = [base_url + str(papyrus_id) for papyrus_id in papyrus_ids]
    return urls

def scrap_papyrus(url):
    # Send a request to the URL
    response = requests.get(url)
    if response.status_code != 200:
        return {"error": "Page not found"}
    
    # Parse the HTML content of the page
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Dictionnary to store the informations
    papyrus_info = {}
    
    # Extract the informations from the page (ID, Date, Provenance, Language/Script, Material, Content)
    Ids = soup.find('div', id='text-header').find('h2').text.strip().split()[:2] if soup.find('div', id='text-header') else 'nan'  
    papyrus_info['ID'] = ' '.join(Ids)
    papyrus_info['Date'] = soup.find('div', class_='division').text.strip().replace("Date:", "") if soup.find('div', class_='division') else 'nan'
    papyrus_info['Provenance'] = soup.find('span', class_='semibold', string='Provenance:').find_next_sibling().text if soup.find('span', class_='semibold', string='Provenance:') else 'nan'
    papyrus_info['Language/Script'] = soup.find('span', class_='semibold', string='Language/script:').find_next_sibling().text if soup.find('span', class_='semibold', string='Language/script:') else 'nan'
    papyrus_info['Material'] = soup.find('span', class_='semibold', string='Material:').find_next_sibling().text if soup.find('span', class_='semibold', string='Material:') else 'nan'
    papyrus_info['Content'] = soup.find('span', class_='semibold', string='Content:').find_next_sibling().text if soup.find('span', class_='semibold', string='Content:') else 'nan'
    
    # Publications
    publications_tag = soup.find('div', id='text-publs')
    papyrus_info['Publications'] = publications_tag.text.strip().replace("\n", " ") if publications_tag else 'nan'
    
    # Collections
    collections_tag = soup.find('div', id='text-coll')
    papyrus_info['Collections'] = collections_tag.text.strip().replace("\n", " ") if collections_tag else 'nan'
    
    # Archive
    archive_tag = soup.find('div', id='text-arch')
    papyrus_info['Archive'] = archive_tag.text.strip().replace("\n", " ") if archive_tag else 'nan'
    
    # Le texte principal
    main_text_tag = soup.find('div', id='words-list')
    papyrus_info['Main Text'] = main_text_tag.text.strip().replace("\n", " ") if main_text_tag else 'nan'
    
    # Irregularités du texte
    text_irregularities_tag = soup.find('div', id='texirr')
    papyrus_info['Text Irregularities'] = text_irregularities_tag.text.strip().replace("\n", " ") if text_irregularities_tag else 'nan'
    
    return papyrus_info

def save_to_csv(data, output_file):
    with open(output_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def main(csv_corpus: csv):
    papyrus_ids = get_papyrus_ids(csv_corpus)
    urls = generate_papyrus_urls(papyrus_ids)
    data = [scrap_papyrus(url) for url in urls[:4]]
    save_to_csv(data, "papyri.csv")

if __name__ == "__main__":
    csv_corpus = "csv/papyrus_corpus.csv"
    main(csv_corpus)