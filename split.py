from langchain.text_splitter import RecursiveCharacterTextSplitter
import glob
import tiktoken
import json
from url_map import url_map

CHUNK_SIZE = 3750

tokenizer = tiktoken.get_encoding("p50k_base")

def tiktoken_len(text):
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)

splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=20, length_function=tiktoken_len)


if __name__ == '__main__':

  chunks = []

  for file in glob.glob("transcripts/*.txt"):
    with open(file, "r") as f:
      text = f.read()

    name = file.replace("transcripts/", "").replace(".txt", "")
    split_text = splitter.split_text(text)
    current_chunks = [
        {
          "text": split_text[i],
          "title": url_map.get(name)["title"],
          "url": url_map.get(name)["url"]
        }
        for i in range(len(split_text))
       ]

    with open(f"chunks/{name}.json", "w") as outfile:
       outfile.write(json.dumps(current_chunks))
    