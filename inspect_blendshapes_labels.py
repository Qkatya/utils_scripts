import pickle

from scipy.signal import dfreqresp

pkl_path = "/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits/loud_and_whisper_and_lip_20250713_064722.pkl"
# pkl_path = "/mnt/ML/Development/ML_Data_DB/v2/splits/full/20251018_split_1/LOUD_GIP_free_speech_question.pkl"
text_to_label_path = "/mnt/ML/ModelsTrainResults/michael.doron/text_to_label_dict.pkl"
with open(text_to_label_path, 'rb') as f:
    text_to_label = pickle.load(f)
with open(pkl_path, 'rb') as f:
    df = pickle.load(f)
df['text_label'] = df.read_text.apply(lambda x: text_to_label[x])
df.to_pickle(pkl_path)
