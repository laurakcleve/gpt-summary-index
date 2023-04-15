import os
import glob
import json
from datetime import datetime
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]


SINGLE_SUMMARY_PROMPT = "The following is a transcription of a video for web developers called '{title}'. Write a thorough and detailed summary of the video, providing an account of the steps taken and topics covered. Include any names that would be useful for a keyword search: names of tools, specific commands or messages, etc. Use the first person point of view, using 'I' instead of referring to the speaker as 'the host' or 'the speaker'.\n\n---\nTRANSCRIPTION:\n\n{transcription}\n\n(END OF TRANSCRIPTION)"

FULL_SUMMARY_PROMPT = "The following are summaries of different parts of a video named '{title}'. Please create a comprehensive and detailed summary that combines all of the individual summaries provided, without omitting any information. The final summary should include all relevant details from each summary and provide a complete overview of the subject matter. It should include any names that would be useful in for a keyword search: names of tools, specific commands or messages, etc. Write it in first person point of view.\n\nSUMMARIES:\n\n{summary_string}"


if __name__ == '__main__':
    for file in glob.glob("chunks/*.json"):
        with open(file, "r") as f:
            chunks = json.load(f)

        file_name = file.replace("chunks/", "").replace(".json", "")
        output_dir = f"summaries/{file_name}"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        title = chunks[0]["title"]

        print(f"Processing {title}...")

        summaries = []

        for i, chunk in enumerate(chunks):
            single_summary_prompt_formatted = SINGLE_SUMMARY_PROMPT.format(title=title, transcription=chunk["text"])
            
            completion = openai.ChatCompletion.create(
                model="gpt-4", 
                temperature=0.7, 
                max_tokens=3000, 
                messages= [{ "role": "user", "content": single_summary_prompt_formatted }]
            )

            summary = completion["choices"][0]["message"]["content"]
            summaries.append(summary)

            timestamp_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            with open(f"{output_dir}/{timestamp_str}_summary_{i}.txt", "w") as file:
                file.write(f"PROMPT:\n\n{single_summary_prompt_formatted}\n\n---\n\nSUMMARY {i}:\n\n{summary}")
            
            print(f"\tFinished summary {i}")

        summary_string = ""
        for s in summaries:
            summary_string += f"{s}\n\n"

        full_summary_prompt_formatted = FULL_SUMMARY_PROMPT.format(title=title, summary_string=summary_string)

        completion = openai.ChatCompletion.create(
            model="gpt-4", 
            temperature=0.7, 
            max_tokens=3000, 
            messages= [{ "role": "user", "content": full_summary_prompt_formatted }]
        )

        final_summary = {
            "summary": completion["choices"][0]["message"]["content"],
            "title": title,
            "url": chunks[0]["url"]
        }

        timestamp_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        with open(f"{output_dir}/{timestamp_str}_final-summary-prompt.txt", "w") as file:
            file.write(full_summary_prompt_formatted)

        with open(f"{output_dir}/{timestamp_str}_final-summary.json", "w") as file:
            file.write(json.dumps(final_summary))

        print(f"\tFinished final summary")