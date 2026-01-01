import os

def load_resume_data():
    folder = "resume"
    combined_text = ""

    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                combined_text += f"\n\n### {file.replace('.txt','').upper()} ###\n"
                combined_text += f.read()

    return combined_text.strip()


RESUME_TEXT = load_resume_data()
