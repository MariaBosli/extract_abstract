from llm import llm
from references import extract_unique_references, extraire_noms_numeros, extraire_references_par_nom_et_numero
from bert import summarize_text
mistral_llm = llm()  # Créez une instance de votre modèle de langage 

def extract_state_of_the_art(abstract):
    # Construction du prompt pour la génération de la demande d'état de l'art
    prompt =f""" [INST] Vous êtes un chercheur académique. Votre tâche est d'ameliorer le paragraphe suivant avec un langage claire et précis. le resultat dois etre dans un seul paragraphe bien rediger.
    paragrapge :
    {abstract}
    """

    # Découpage du prompt en chunks pour respecter la limite de longueur
    max_length = 512
    prompt_chunks = [prompt[i:i+max_length] for i in range(0, len(prompt), max_length)]

    # Génération du texte de demande d'état de l'art à partir du prompt en utilisant le modèle de langage mixte
    generated_text_chunks = []
    for prompt_chunk in prompt_chunks:
        output = mistral_llm(prompt_chunk)
        generated_text_chunks.append(output)

    generated_text = ''.join(generated_text_chunks)
    generated_text = '\n'.join([p for p in generated_text.split('\n')[1:] if len(p) > 0])

    # Retour du texte généré
    return generated_text

def main():
    #Appeler la fonction bert sum pour faire le resumer des abstract avant de promper mixtral
    body='''Since late December 2019, a new coronavirus
    outbreak of Covid-19 has been recorded in Wuhan, China, and
    then became epidemic all over the world. The onset of Covid-19
    can result in death as a result of important alveolar damage and

    advancing respiratory insufficiency. Though the transcription-
    polymerase chain reaction (RT-PCR) used for clinical

    determination is the gold standard, tests can obvious false
    negatives. In addition, in the pandemic situation, insufficient
    RT-PCR test resources may delay diagnosis and treatment.
    Under these circumstances, Computed Tomography (CT) scans
    have become a precious vehicle for both early diagnosis and
    prognosis of Covid-19 patients. Recently, many studies
    developed with deep learning techniques have been proffered to
    facilitate the diagnosis of Covid-19 in CT scans and to assist
    healthcare professionals. The purpose of this article is to first
    create a mixed dataset by coloring some of the CT images with
    the DeOldify method in order to make a more right
    performance and then to detect COVID-19 cases using
    DenseNet121, one of the deep learning (DL) techniques.
    Abstract—The COVID-19 pandemic is one of the most
    challenging healthcare crises during the 21st century. As the virus
    continues to spread on a global scale, the majority of efforts have
    been on the development of vaccines and the mass immunization
    of the public. While the daily case numbers were following a
    decreasing trend, the emergent of new virus mutations and
    variants still pose a significant threat. As economies start
    recovering and societies start opening up with people going back
    into office buildings, schools, and malls, we still need to have the
    ability to detect and minimize the spread of COVID-19.
    Individuals with COVID-19 may show multiple symptoms such as
    cough, fever, and shortness of breath. Many of the existing
    detection techniques focus on symptoms having the same equal
    importance. However, it has been shown that some symptoms are
    more prevalent than others. In this paper, we present a
    multimodal method to predict COVID-19 by incorporating
    existing deep learning classifiers using convolutional neural
    networks and our novel probability-based weighting function that
    considers the prevalence of each symptom. The experiments were
    performed on an existing dataset with respect to the three
    considered modes of coughs, fever, and shortness of breath. The

    results show considerable improvements in detection of COVID-
    19 using our weighting function when compared to an equal

    weighting function.
    Abstract—Deep Learning has improved multi-fold in recent

    years and it has been playing a great role in image clas-
    sification w hich a lso i ncludes m edical i maging. Convolutional

    Neural Networks (CNN) has been performing well in detecting
    many diseases including Coronary Artery Disease, Malaria,
    Alzheimer’s disease, different dental diseases, and Parkinson’s
    disease. Like other cases, CNN has a substantial prospect in

    detecting COVID-19 patients with medical images like chest X-
    rays and CTs. Coronavirus or COVID-19 has been declared a

    global pandemic by the World Health Organization (WHO). Till
    July 11, 2020, the total COVID-19 confirmed c ases a re 1 2.32 M
    and deaths are 0.556 M worldwide. Detecting Corona positive
    patients is very important in preventing the spread of this virus.
    On this conquest, a CNN model is proposed to detect COVID-19
    patients from chest X-ray images. This model is evaluated with
    a comparative analysis of two other CNN models. The proposed
    model performs with an accuracy of 97.56% and a precision of
    95.34%. This model gives the Receiver Operating Characteristic
    (ROC) curve area of 0.976 and F1-score of 97.61. It can be
    improved further by increasing the dataset for training the model.'''

    
    abstract = summarize_text(body)
    # Appel de la fonction extract_state_of_the_art avec la liste de travaux connexes
    etat_art = extract_state_of_the_art(abstract)

    # Affichage du résultat
    print(etat_art)
    


if __name__ == "__main__":
    main()