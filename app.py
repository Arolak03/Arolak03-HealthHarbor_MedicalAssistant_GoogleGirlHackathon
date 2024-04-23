import pandas as pd
import PIL
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import csv
from PIL import Image
import os

# Initialize an empty dictionary to store the mapping
specialist_map = {}

# Read the CSV file and populate the dictionary
with open('diseases.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        key = row[0]
        values = row[1:]
        specialist_map[key] = values


free_slot_map = {}
with open('doctor_schedule.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        key = row[0]
        free_slots = [slot for slot in row[1:] if slot.strip().lower() == 'free']  # Extracting free slots
        free_slots_with_time = [(slot, index) for index, slot in enumerate(row[1:], start=1) if slot.strip().lower() == 'free']
        # Create a list to store the time headings
        time_headings = ['8am - 9am', '9am - 10am', '10am - 11am', '11am - 12pm', '12pm - 1pm', '1pm - 2pm', '2pm - 3pm', '3pm - 4pm', '4pm - 5pm', '5pm - 6pm']
        # Filter the time headings based on the indices of the free slots
        free_slot_times = [time_headings[index-1] for slot, index in free_slots_with_time]
        # print(free_slot_times)
        # Store the free slots and their corresponding times in the free_slot_map
        free_slot_map[key] = free_slot_times


def display_values_as_list(key):
    if key in specialist_map:
        values = specialist_map[key]
        html_output = "<ul>"
        for value in values:
            html_output += f"<li>{value}</li>"
        html_output += "</ul>"
        st.markdown(html_output, unsafe_allow_html=True)
    else:
        st.write("Key not found")


def display_values_as_list_schedule(key):
    if key in free_slot_map:
        print(key)
        values=free_slot_map[key]
        html_output = "<ul>"
        for value in values:
            print(values)
            html_output += f"<li>{value}</li>"
        html_output += "</ul>"
        st.markdown(html_output, unsafe_allow_html=True)
    else:
        st.write("Key not found")

# Load the doctor dataset from CSV
doctors_df = pd.read_csv('doctors.csv')

# Load the symptoms and specialist dataset from CSV
symptoms_specialist_df = pd.read_csv('output.csv')


def disease_linked(Specialist):
    st.write(specialist_map[Specialist])

def display_profile(row):
    col1, col2 = st.columns([1, 1])
    # Content for the first column
    image_folder = "Images"
    with col1:
        # st.image(row['Profile Picture'], caption=row['Name'], use_column_width=True)
        image_path = os.path.join(image_folder, row['Profile Picture'])
        # Check if the file exists
        if os.path.exists(image_path):
            # Display the image
            st.image(image_path, caption=row['Name'], use_column_width=True)

    with col2:
        st.write(f"**Specialty:** {row['Specialty']}")
        st.write(f"**Location:** {row['Location']}")
        st.write(f"**Experience:** {row['Experience']}")
        st.write(f"**Ratings:** {row['User Rating']}")

    col3,col4 = st.columns([1,1])
    with col3:
        st.write(f"**Diseases Treating**")
        display_values_as_list(row['Specialty'])
        
    with col4:
        st.write(f"**Free Slots available**")
        display_values_as_list_schedule(row['Name'])
    # Content for the second column
    # with col2:
        
# Application Frontend
st.title('Health Harbor')

# Collect user symptoms
selected_symptoms = st.multiselect('Select your symptoms:', symptoms_specialist_df['symptoms'].unique())

# Filter specialist based on selected symptoms
filtered_specialist = symptoms_specialist_df[symptoms_specialist_df['symptoms'].isin(selected_symptoms)]


if filtered_specialist['specialist'].nunique() < 2:
    st.write('Insufficient data to make a recommendation.')
else:
    # Train the classifier
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(filtered_specialist['symptoms'])
    y = filtered_specialist['specialist']
    classifier = LinearSVC()
    classifier.fit(X, y)

    # Predict specialist for the selected symptoms
    X_user = vectorizer.transform(selected_symptoms)
    predicted_specialist = classifier.predict(X_user)

    # Filter doctors based on predicted specialist
    filtered_doctors = doctors_df[doctors_df['Specialty'].isin(predicted_specialist)]

    # Display recommended doctors
    if not filtered_doctors.empty:
        st.subheader('Recommended Doctors:')
        # Sort doctors by user rating in descending order
        filtered_doctors_sorted = filtered_doctors.sort_values(by='User Rating', ascending=False)
        for index, row in filtered_doctors_sorted.iterrows():
            with st.expander(f"{row['Name']}"):
                display_profile(row)
            st.write(f"**Specialty:** {row['Specialty']}")
            st.write(f"**Location:** {row['Location']}")
            st.write(f"**Experience:** {row['Experience']}")
            st.write(f"**Ratings:** {row['User Rating']}")
            st.write("---")

    else:
        st.write('No doctors found for the selected symptoms.')

# Count the occurrences of each symptom in the filtered specialist DataFrame
specialist_counts = filtered_specialist['specialist'].value_counts()

# st.sidebar.subheader('Contact Me', expanded=False)
# st.sidebar.write('Developer Name: Palak Arora')
# st.sidebar.write('Contact me at 2021ucs0105@iitjammu.ac.in | 7089851331')
# # st.sidebar.write('Contact Me at ')
# st.sidebar.header('Contact Me')

#     # Content for the "Contact Me" section
# # 
# st.sidebar.write("Palak Arora")
# st.sidebar.write("Email: palkarora1298@gmail.com")
# st.sidebar.write("Phone: 7089851331")

with st.sidebar:
    st.header('**Contact Me**')
    st.subheader("Palak Arora")
    st.write("Email: palkarora1298@gmail.com")
    st.write("Phone: 7089851331")

# Plot the bar graph
plt.figure(figsize=(10, 6))
plt.bar(specialist_counts.index, (specialist_counts.values)*100/len(selected_symptoms), color='skyblue')
plt.xlabel('Specialists')
plt.ylabel('Percentage(%) of Symptoms matching')
plt.title('Relevance of Symptoms with Specialists')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Format y-axis to display percentage symbol along with digits
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.tight_layout()

# Display the bar graph in the Streamlit app
st.pyplot(plt)

selected_specialty = st.selectbox('Select a specialty:', doctors_df['Specialty'].unique())

# Filter doctors based on selected specialty
filtered_doctors_by_specialty = doctors_df[doctors_df['Specialty'] == selected_specialty]

# Display doctors by specialty
if not filtered_doctors_by_specialty.empty:
    st.subheader(f'{selected_specialty}:')
    # Sort doctors by user rating in descending order
    filtered_doctors_sorted = filtered_doctors_by_specialty.sort_values(by='User Rating', ascending=False)
    for index, row in filtered_doctors_sorted.iterrows():
        with st.expander(f"{row['Name']}"):
            display_profile(row)
        st.write(f"**Specialty:** {row['Specialty']}")
        st.write(f"**Location:** {row['Location']}")
        st.write(f"**Experience:** {row['Experience']}")
        st.write(f"**Ratings:** {row['User Rating']}")
        st.write("---")
else:
    st.write(f'No doctors found specialized in {selected_specialty}.')
    st.write("Displaying General Physicians:")
    general_physicians = doctors_df[doctors_df['Specialty'] == 'General Physician']
    for index, row in general_physicians.iterrows():
        with st.expander(f"{row['Name']}"):
            display_profile(row)
        st.write(f"**Specialty:** {row['Specialty']}")
        st.write(f"**Location:** {row['Location']}")
        st.write(f"**Experience:** {row['Experience']}")
        st.write("---")

# Add custom CSS
css='''
<style>
h1, h2, h3, h4, h5, h6 {
    color: #f0e995 !important;
    padding: 10px 

    font-family: 'Arial', sans-serif;
}

body {
    padding: 20px;
}

.card {
    background-color: #ffffff;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    padding: 20px;
    margin-bottom: 20px;
    border: solid !important;
}

.card-title {
    font-size: 20px;
    font-weight: bold;
    margin-bottom: 10px;
}
div[data-testid="stExpander"] details summary {
    padding: 10px; /* Adjust the value as needed */
}

div[data-testid="stExpander"] details summary p {
    font-size: 1.5rem;
    
    color: #f05561 !important;

    padding: 5px; /* Adjust the value as needed */
}


.card-content {
    font-size: 16px;
    margin-bottom: 10px;
}

hr {
    border: none;
    border-top: 1px solid #ccc;
    margin: 20px 0;
}
expander{
    color: red;
}

</style>
'''
st.markdown(css, unsafe_allow_html=True)
# st.markdown(html, unsafe_allow_html=True)
