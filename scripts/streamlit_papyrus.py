import streamlit as st
import pandas as pd
import re
import folium
from difflib import ndiff
from streamlit_folium import folium_static
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import GridUpdateMode 


# Charger les données
collection = pd.read_csv("./result/clean_papyrus-corpus.csv")

# Fonction pour extraire l'année de début à partir de la colonne 'Date'
def extract_year(date_str):
    match = re.search(r'AD (\d{1,4})', date_str)
    if match:
        return int(match.group(1))
    else:
        return None

# Appliquer la fonction pour créer une nouvelle colonne 'Year'
collection['Year'] = collection['Date'].apply(extract_year)

# Supprimer les lignes avec des années non valides
collection = collection.dropna(subset=['Year'])

# Gérer la navigation entre les pages avec `st.session_state`
if 'selected_papyrus_id' not in st.session_state:
    st.session_state['selected_papyrus_id'] = None

# Fonction pour extraire les différences entre deux mots et les colorer
def highlight_word_differences(word1, word2):
    diff = ndiff(word1, word2)
    highlighted_text = ""

    for char in diff:
        if char.startswith("-"):
            # Texte supprimé (rouge)
            highlighted_text += f'<span style="color:red;">{char[2]}</span>'
        elif char.startswith("+"):
            # Texte ajouté (vert)
            highlighted_text += f'<span style="color:green;">{char[2]}</span>'
        else:
            # Texte inchangé
            highlighted_text += char[2]

    return highlighted_text

# Fonction pour traiter les irrégularités textuelles et les afficher avec des différences en couleur
def highlight_differences(text):
    # Séparer par virgule pour obtenir chaque paire de mots
    irregularities = text.split(", ")
    highlighted_irregularities = []

    for item in irregularities:
        if ":" in item:
            # Cas où nous avons une paire à comparer (mot1: mot2)
            word1, word2 = item.split(": ")
            # Appliquer la coloration
            highlighted_pair = f'{highlight_word_differences(word1.strip(), word2.strip())}'
            highlighted_irregularities.append(f'{word1}: {highlighted_pair}')
        else:
            # En cas de format incorrect, on ajoute sans modification
            highlighted_irregularities.append(item.strip())

    return ", ".join(highlighted_irregularities)

# Fonction pour afficher les détails du papyrus
def show_papyrus_details(papyrus_id):
    papyrus = collection[collection["ID"] == papyrus_id]
    
    # Affichage du numéro de papyrus
    st.write(f"**Papyrus n°{papyrus_id}**")
    
    # Affichage de l'auteur
    auteur = papyrus['Author'].values[0]
    if pd.isna(auteur) or auteur == "" or auteur == "nan":
        auteur = "Non spécifié"
    st.write(f"Auteur: <br> {auteur}", unsafe_allow_html=True)

    # Affichage de la date (aucune modification nécessaire)
    st.write(f"Date: {papyrus['Date'].values[0]}")
    
    # Gestion des valeurs vides pour le genre
    genre = papyrus['Genre'].values[0]
    if pd.isna(genre) or genre == "":
        genre = "Non spécifié"
    st.write(f"Genre:<br> {genre}", unsafe_allow_html=True)

    # Gestion des valeurs vides pour la provenance
    provenance = papyrus['Provenance'].values[0]
    if pd.isna(provenance) or provenance == "":
        provenance = "\n Non spécifié"
    st.write(f"Provenance: <br> {provenance}", unsafe_allow_html=True)
    
    # Gestion des valeurs vides pour les personnes
    personnes = papyrus['Personnes'].values[0]
    if pd.isna(personnes) or personnes == "":
        personnes = "Non spécifié"
    else:
        personnes = ", ".join(personnes.replace("'", "").split(","))
    st.write(f"Personnes: <br> {personnes}", unsafe_allow_html=True)
    
    # Gestion des valeurs vides pour les lieux
    lieux = papyrus['Lieux'].values[0]
    if pd.isna(lieux) or lieux == "nan" or lieux == "":
        lieux = "Non spécifié"
    else:
        lieux = ", ".join(lieux.replace("'", "").replace("[", "").split(","))
    st.write(f"Lieux: <br> {lieux}", unsafe_allow_html=True)
    
    # Gestion des valeurs vides pour les irrégularités textuelles
    text_irregularities = papyrus['Text Irregularities'].values[0]
    if pd.isna(text_irregularities) or text_irregularities == "[]" or text_irregularities == "":
        text_irregularities = "Aucune"
    else:
        text_irregularities = ", ".join(text_irregularities.replace("'", "").replace("[", "").replace("]", "").split(","))
        # Utiliser st.markdown avec un texte stylisé pour afficher les irrégularités colorées
        st.markdown(f"Textual Irregularities (rouge : suppression ; vert : ajout):<br> {highlight_differences(text_irregularities)}", unsafe_allow_html=True)

    # Retourner la provenance pour l'affichage sur la carte
    return provenance

# Créer des onglets pour la navigation
tab1, tab2 = st.tabs(["Accueil", "Papyri"])

# Contenu de l'onglet "Accueil"
with tab1:
    st.title("La Chasse aux Papyrus")
    
    # Centrer l'image uniquement sur la page d'accueil
    st.image("./Book_of_the_Dead_of_Hunefer_sheet_3.jpg", 
             caption="Book of the Dead of Hunefer", width=400, use_container_width=True)

    st.write("Cette application permet de découvrir les papyrus. Vous pouvez choisir un papyrus dans le menu déroulant à gauche. \
          Vous pouvez également filtrer les papyrus par provenance, date de découverte, personne et lieu.")

    st.header("Carte des Provenances")

    # Ajouter une carte Folium centrée sur l'Égypte
    m = folium.Map(location=[26.696361500000002,30.246486000000004], zoom_start=5, control_scale=True)

    # Dictionnaire des lieux avec leurs coordonnées géographiques
    locations = {
        "Ibion": [29.18493, 30.66422],
        "Aphrodito": [31.25167, 29.40842],
        "Antinoopolis": [30.8793, 27.8081],
        "Byzantion": [28.97593, 41.01238],
        "Alexandria": [29.90413, 31.19537],
        "Hermopolites": [27.7756214,30.8047827],
        "Arabon Kome": [26.696361500000002,30.246486000000004],
        "Egypt": [26.696361500000002,30.246486000000004],
        "Lykopolites": [27.181346,31.1825335],
        "Phthla": [31.42469, 26.92388],
        "Tanyaithis": [26.696361500000002,30.246486000000004],
        "Zmin": [26.696361500000002,30.246486000000004],
        "Antaiopolis" : [26.8992835,31.517443],
    }

    # Ajouter des marqueurs pour chaque lieu sur la carte
    for lieu, coord in locations.items():
        folium.Marker(coord, popup=lieu).add_to(m)

    # Afficher la carte avec folium_static
    folium_static(m)

# Contenu de l'onglet "Papyri"
with tab2:
    st.sidebar.header("Filtres")

    # Filtres pour la provenance avec l'option "Tous"
    lieux = list(collection["Provenance"].dropna().unique())
    lieux.insert(0, "Tous")
    lieu = st.sidebar.selectbox("Filtrer par provenance", lieux)

    # Slider pour la plage de dates
    date_range = st.sidebar.slider(
        "Filtrer par date de découverte",
        min_value=int(collection['Year'].min()),
        max_value=int(collection['Year'].max()),
        value=(int(collection['Year'].min()), int(collection['Year'].max()))
    )

    # Appliquer les filtres de provenance et de date
    filtered_collection = collection.copy()

    if lieu != "Tous":
        filtered_collection = filtered_collection[filtered_collection["Provenance"] == lieu]

    filtered_collection = filtered_collection[
        (filtered_collection["Year"] >= date_range[0]) &
        (filtered_collection["Year"] <= date_range[1])
    ]

    # Nettoyer les colonnes Personnes et Lieux
    person_list = filtered_collection["Personnes"].dropna().apply(
        lambda x: [person.strip("' ").strip() for person in x.split(",")]
    ).tolist()
    person_list = [person for sublist in person_list for person in sublist if person]  # Aplatir la liste et retirer les vides
    person_list = sorted(set(person_list))  # Trier et dédupliquer
    person_list.insert(0, "Tous")

    lieu_list = filtered_collection["Lieux"].dropna().apply(
        lambda x: [lieu.strip("' ").strip() for lieu in x.split(",")]
    ).tolist()
    lieu_list = [lieu for sublist in lieu_list for lieu in sublist if lieu]  # Aplatir la liste et retirer les vides
    lieu_list = sorted(set(lieu_list))  # Trier et dédupliquer
    lieu_list.insert(0, "Tous")

    # Sélection de la personne et du lieu après avoir appliqué les filtres dynamiques
    person = st.sidebar.selectbox("Filtrer par personne", person_list)
    lieu_recherche = st.sidebar.selectbox("Filtrer par lieu", lieu_list)

    # Appliquer le filtre des personnes
    if person != "Tous":
        filtered_collection = filtered_collection[
            filtered_collection["Personnes"].str.contains(person, case=False, na=False)
        ]

    # Appliquer le filtre des lieux
    if lieu_recherche != "Tous":
        filtered_collection = filtered_collection[
            filtered_collection["Lieux"].str.contains(lieu_recherche, case=False, na=False)
        ]


    # Afficher le tableau interactif
    st.write("### Liste complète des papyrus")

    gb = GridOptionsBuilder.from_dataframe(
        filtered_collection[['ID', 'Author', 'Date', 'Genre', 'Provenance', 'Personnes', 'Lieux']]
    )
    gb.configure_selection('single', use_checkbox=True)  # Ajout d'une checkbox pour la sélection
    grid_options = gb.build()

    grid_response = AgGrid(
        filtered_collection[['ID', 'Author', 'Date', 'Genre', 'Provenance', 'Personnes', 'Lieux']],
        gridOptions=grid_options,
        height=400,
        fit_columns_on_grid_load=True,
        enable_enterprise_modules=False,
        key='grid',
        update_mode=GridUpdateMode.SELECTION_CHANGED  # Mise à jour uniquement lors des changements de sélection
    )

    # Récupérer les lignes sélectionnées
    selected_rows = grid_response['selected_rows']
    
    # Vérifier si des lignes sont sélectionnées
    if isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty:
        # Récupérer la première ligne sélectionnée
        selected_row = selected_rows.iloc[0]
        
        # Extraire l'ID
        selected_id = selected_row['ID']
        
        # Afficher les détails du papyrus
        selected_provenance = show_papyrus_details(selected_id)
        
        # Afficher la carte si une provenance est trouvée
        if selected_provenance in locations:
            st.write("**Carte de l'Egypte**")
            m = folium.Map(location=locations[selected_provenance], zoom_start=6)
            folium.Marker(locations[selected_provenance], popup=selected_provenance).add_to(m)
            folium_static(m)
    else:
        st.info("Aucune ligne sélectionnée")